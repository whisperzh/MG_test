# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import torch.distributed as dist
from megatron.core import mpu
from torch.distributed import get_rank
import copy
import os
import torch
import numpy as np
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        if int(os.getenv("DEBUG", "0")) == 1:
            print(f"[Rank {get_rank()}] Local experts: {self.num_local_experts}, Indices: {self.local_expert_indices}")
        # Initialize router, the router can not be modified, because router is pretrained 
        self.router = TopKRouter(config=copy.deepcopy(self.config))
        # NOTE This must be called after initializing self.router. Otherwise router will be influenced.
        self.ep_rank = mpu.get_expert_model_parallel_rank()
        self.ep_world_size =  mpu.get_expert_model_parallel_world_size()
        if int(os.getenv("EPLB", "0")) == 1:
            self.init_eplb()
        if int(os.getenv("REPLICATE", "0")) == 1:
            # replciate on expert in each rank
            self.init_replicate()
        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        
        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)
    def init_replicate(self):
        print("Replicate one expert in each rank")
        self.num_local_experts += 1 
        self.config.num_moe_experts += mpu.get_expert_model_parallel_world_size()
        # NOTE: currently, self.local_expert_indices must be continious
        # Original local experts per rank:     [0,1], [2,3], [4,5], [6,7]
        # After replication (1 additional expert per rank):  [0,1,2] [3,4,5] [6,7,8] [9,10,11]
        expert_indices = list(range(self.config.num_moe_experts))
        start_idx =  self.ep_rank *  self.num_local_experts
        end_idx =  start_idx + self.num_local_experts
        self.local_expert_indices  = expert_indices[start_idx:end_idx]
        new_id_2_old_id_map = dict()
        # Mapping from new expert index to its source (replicated) expert
        # Currently, each newly added expert replicates the first local expert on the same rank
        for i in range(0, mpu.get_expert_model_parallel_world_size()):
            old_id = i *  self.num_local_experts
            new_id = old_id +  self.num_local_experts -1 
            new_id_2_old_id_map[new_id] = old_id
        self.new_id_2_old_id_map = new_id_2_old_id_map
        print(f"[RANK {self.ep_rank}] Update ...")
        print("num_moe_experts",self.config.num_moe_experts)
        print("local_expert_indices", self.local_expert_indices)
        print("new_id_2_old_id_map",new_id_2_old_id_map)
        print("==================")
    def init_eplb(self):
        assert os.getenv("EPLB") == "1"
        self.eplb_para = dict()
        self.eplb_para["ep_rank"] = mpu.get_expert_model_parallel_rank()
        self.eplb_para["ep_world_size"] = mpu.get_expert_model_parallel_world_size()
        self.eplb_para["num_replica"] =  self.eplb_para["ep_world_size"]
        self.eplb_para["num_nodes"] = 1
        self.eplb_para["num_groups"] =  self.config.num_moe_experts
        self.eplb_para["num_gpus"] = self.eplb_para["ep_world_size"]
        self.eplb_para["replicated_expert_per_rank"] = (self.eplb_para["num_replica"]) //  self.eplb_para["ep_world_size"]
        #TODO: required modify for multi node 
        base_expert_id = self.config.num_moe_experts
        # 
        self.num_local_experts += self.eplb_para["replicated_expert_per_rank"]
        self.config.num_moe_experts += self.eplb_para["num_replica"]
        expert_indices = list(range(self.config.num_moe_experts))
        start_idx =  self.ep_rank *  self.num_local_experts
        end_idx =  start_idx + self.num_local_experts
        self.local_expert_indices  = expert_indices[start_idx:end_idx]

            
        print(f"RANK[{ self.ep_rank}] :")
        print("\n[EPLB CONFIGURATION]")
        for k, v in self.eplb_para.items():
            print(f"  {k}: {v}")
        print(self.num_local_experts)
        print(self.local_expert_indices)
        print()
    def eplb(self,routing_map):
        raise NotImplementedError 
        assert os.getenv("EPLB") == "1", "EPLB is not enabled"
        # token for each expert,routing_map is the original map, for mixtral size = [TOKEN_NUM, 8]
        workload_distribution = routing_map.sum(dim=0).long() # for the original expert 0 - 7
        assert self.workload_distribution.ndim == 2, "workload_distribution must be 2D"
        assert self.workload_distribution.shape[0] == 1, "The first dimension of workload_distribution must be 1"
        phy2log, log2phy, _ = rebalance_experts(self.workload_distribution ,  
                                                self.global_num_experts + self.eplb_para["num_replica"] , 
                                                self.eplb_para["num_groups"], 
                                                self.eplb_para["num_nodes"], 
                                                self.eplb_para["num_gpus"])
        # get new auxiliary_expert_map
        self.auxiliary_expert_map = dict()
        new_idx = self.global_num_experts
        new_phy2log = copy.deepcopy(phy2log)
        for i in range(log2phy.shape[1]):
            for j in range(1,log2phy.shape[2]):
                if (log2phy[0][i][j].item() != -1):
                    # print("replicate ", i, "new_id", new_idx)
                    pos = log2phy[0][i][j].item()
                    self.auxiliary_expert_map[ new_idx] = i
                    new_phy2log [0][pos] = new_idx
                    new_idx += 1
        id_2_new_id_map = dict()
        for new_id, old_id in self.auxiliary_expert_map.items():
            if id_2_new_id_map.get(old_id) is None:
                id_2_new_id_map[old_id] = [new_id]
            else:
                id_2_new_id_map[old_id].append(new_id)
        self.id_2_new_id_map = id_2_new_id_map
        # print(self.auxiliary_expert_map )
        # get new pros
        # get new routing_map
        # get new weight pass currently 
        
    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        # process MoE
        def custom_forward(hidden_states):
            rank = dist.get_rank()
            if int(os.getenv("DEBUG", "0")) == 1:
                print(f"RANK[{rank}] : hidden_states.shape = {hidden_states.shape}")
            if int(os.getenv("REPLICATE", "0")) == 1:
                # === Expert Weight Replication Phase ===
                # This step copies the weights from an existing expert to a newly added replicated expert
                print(f"[RANK {self.ep_rank}] Update New expert weight")
                # Get global expert ID of the newly added expert (last one in local list)
                new_id = self.local_expert_indices[-1]
                # Get the corresponding old expert (the one being replicated)
                old_id = self.new_id_2_old_id_map[new_id]
                # Convert global expert IDs to local offset within the rank
                offset_new_id =  new_id % self.num_local_experts 
                offset_old_id = old_id % self.num_local_experts 
                # --- Copy weights for linear_fc1 ---
                new_attr_name = f"weight{offset_new_id}"
                old_attr_name = f"weight{offset_old_id}"
                new_expert_weight = getattr(self.experts.linear_fc1, new_attr_name)
                old_expert_weight = getattr(self.experts.linear_fc1, old_attr_name)
                with torch.no_grad():
                    new_expert_weight.data.copy_(old_expert_weight.data)
                # --- Copy weights for linear_fc2 ---
                new_attr_name = f"weight{offset_new_id}"
                old_attr_name = f"weight{offset_old_id}"
                new_expert_weight = getattr(self.experts.linear_fc2, new_attr_name)
                old_expert_weight = getattr(self.experts.linear_fc2, old_attr_name)
                with torch.no_grad():
                    new_expert_weight.data.copy_(old_expert_weight.data)
            probs, routing_map = self.router(hidden_states)
            if int(os.getenv("IDEAL", "0")) == 1:
                # create ideal routing_map
                idel_routing_map,_ = generate_balanced_routing_map(
                    token_num = hidden_states.shape[0]*hidden_states.shape[1],
                    num_experts = self.config.num_moe_experts ,
                    topk = self.config.moe_router_topk,
                    device =  hidden_states.device 
                )
                routing_map = idel_routing_map
            if int(os.getenv("SKEW", "0")) == 1:
                # create imbalanced routing_map
                get_imbalanced_routing_map(
                    routing_map, expert_id=0, enforce_row_count= 1000
                )
            if int(os.getenv("REPLICATE", "0")) == 1:
                # === Workload Migration Phase for Expert Replication ===
                # Expert layout:
                #   Original expert layout:         [0,1], [2,3], [4,5], [6,7]
                #   After replication (1 expert/rank): [0,1,2], [3,4,5], [6,7,8], [9,10,11]
                # Routing tensors before modification:
                #   routing_map shape: [TOKEN_NUM, 8]
                #   probs shape:       [TOKEN_NUM, 8]
                # After replicate_modify:
                #   routing_map shape: [TOKEN_NUM, 12]
                #   probs shape:       [TOKEN_NUM, 12]
                print("original token per expert", routing_map.sum(dim=0).long() )
                # Expand routing_map and probs to include replicated expert columns (zeros added)
                routing_map = replicate_modify(
                    routing_map,self.ep_world_size , 1
                )
                probs = replicate_modify (
                    probs,self.ep_world_size  ,1
                )
                print("routing_map", routing_map.shape)
                print("probs", probs.shape)
                print("token per expert", routing_map.sum(dim=0).long() )
                # === Migrate token assignments from old expert to new replicated expert ===
                for new_id, old_id in self.new_id_2_old_id_map.items():
                    # Find all token positions assigned to old expert
                    idxs = (routing_map[:, old_id]).nonzero(as_tuple=True)[0]
                    # Select half of the tokens to reroute to the new expert
                    half = idxs[:idxs.size(0) // 2]
                    # Update routing_map: remove from old expert, add to new expert
                    routing_map[half, old_id] = False
                    routing_map[half, new_id] = True
                    # Migrate probability mass to new expert
                    probs[half, new_id] = probs[half, old_id]
                    probs[half, old_id] = 0
                print("update token per expert", routing_map.sum(dim=0).long() )
            ##############################################################
            if int(os.getenv("MOE_TIME", "0")) == 1:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            ##############################################################
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            print(f"RANK[{rank}] dispatched_input \n",dispatched_input.shape)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)
            ##############################################################
            if int(os.getenv("MOE_TIME", "0")) == 1:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                print(f"RANK[{rank}] moe layer elapsed {elapsed} ms\n",)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)
        # if int(os.getenv("REPLICATE", "0")) == 0:
        #     torch.save(output, f"/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/REPLCATE/{get_rank()}.pt")
        # else:
        #     torch.save(output, f"/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/REPLCATE/{get_rank()}_REPLICATE.pt")
        return output, mlp_bias

def generate_balanced_routing_map(token_num, num_experts, topk,device):
    assert topk <= num_experts, "topk must be ≤ num_experts"

    routing_map = torch.zeros((token_num, num_experts), dtype=torch.bool)
    expert_counts = np.zeros(num_experts, dtype=int)

    for i in range(token_num):
        # 从当前计数最小的 experts 中选择 topk 个
        topk_experts = np.argsort(expert_counts)[:topk]
        routing_map[i, topk_experts] = True
        expert_counts[topk_experts] += 1
    routing_map = routing_map.to(device)
    return routing_map, expert_counts
def get_imbalanced_routing_map(routing_map: torch.Tensor, expert_id: int, enforce_row_count: int):
    # modify routing_map in-place
    token_num, num_experts = routing_map.shape
    assert 0 <= expert_id < num_experts
    if enforce_row_count > token_num:
        enforce_row_count = token_num
    for i in range(enforce_row_count):
        row = routing_map[i]
        if not row[expert_id]:
            # 找出当前为 True 的 expert 中的一个，排除 expert_id（避免替换到它）
            true_indices = row.nonzero(as_tuple=True)[0]
            replace_idx = true_indices[torch.randint(len(true_indices), (1,)).item()]
            row[replace_idx] = False
            row[expert_id] = True
            routing_map[i] = row  # 写回
    return routing_map

 
def replicate_modify(data: torch.Tensor, world: int, replicated_num:int, ) -> torch.Tensor:
    """
    Split data along the last dim into `world` chunks, then pad `replicated_num` columns
    (filled with zeros) to each chunk, and concatenate back.

    Args:
        data (Tensor): Input tensor of shape [NUM, num_expert].
        world (int): Number of chunks to split along dim=1.
        replicated_num (int): Number of columns to pad (per chunk) with zeros.

    Returns:
        Tensor: Output shape [NUM, num_expert + world * replicated_num]
    """
    if replicated_num < 0:
        raise ValueError("replicated_num must be >= 0")
    NUM, num_expert = data.shape
    assert num_expert % world == 0, "num_expert must be divisible by world"
    # Step 1: split
    chunks = torch.chunk(data, world, dim=1)  # list of [NUM, num_expert // world]
    # Step 2: pad each with one column of False / 0
    padded_chunks = [
        torch.cat([chunk, torch.zeros((NUM, replicated_num), dtype=data.dtype, device=data.device)], dim=1)
        for chunk in chunks
    ]
    # Step 3: concatenate all chunks back
    new_data = torch.cat(padded_chunks, dim=1)  # shape: [NUM, num_expert + world * replicated_num]
     
    return new_data

