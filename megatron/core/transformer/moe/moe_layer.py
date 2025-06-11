# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import torch.distributed as dist
from megatron.core import mpu
from torch.distributed import get_rank
import copy
from collections import Counter
import os
import re
from .eplb import rebalance_experts
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
import json
import pickle


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
            self.auxiliary_cpu_experts_weight = None
           
            
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
        # DEBUG
        new_id_2_old_id_map = dict()
        # Mapping from new expert index to its source (replicated) expert
        # Currently, each newly added expert replicates the first local expert on the same rank
        for i in range(0, mpu.get_expert_model_parallel_world_size()):
            old_id = i *  self.num_local_experts
            new_id = old_id +  self.num_local_experts -1 
            new_id_2_old_id_map[new_id] = old_id
        self.new_id_2_old_id_map = new_id_2_old_id_map
        self.replicate_weights_ready = False
        print(f"[RANK {self.ep_rank}] Update ...")
        print("num_moe_experts",self.config.num_moe_experts)
        print("local_expert_indices", self.local_expert_indices)
        print("new_id_2_old_id_map",new_id_2_old_id_map)
        print("==================")
    def init_eplb(self):
        print("EPLB: one auxiliary expert in each rank")
        assert os.getenv("EPLB") == "1"
        self.eplb_para = dict()
        self.eplb_para["ep_rank"] =  self.ep_rank
        self.eplb_para["ep_world_size"] = self.ep_world_size 
        # Number of auxiliary (replicated) experts per rank (currently, fixed as 1 in EPLB)
        self.eplb_para["replicated_expert_per_rank"] = 1
        # Total number of auxiliary experts in the system
        self.eplb_para["num_replica"] =  self.ep_world_size  * self.eplb_para["replicated_expert_per_rank"]
        # Number of nodes (currently assumed to be 1; modify if multi-node is supported)
        # TODO: modify if multi-node support is needed
        self.eplb_para["num_nodes"] = 1
        # Number of original expert groups (before adding auxiliary experts)
        self.eplb_para["num_groups"] =  self.config.num_moe_experts #TODO: for other models, maybe different 
        self.eplb_para["num_gpus"] =  self.ep_world_size 
        # Save the original global expert indices before adding auxiliary experts
        self.original_global_expert_indices  =  list(range(self.config.num_moe_experts))
        # Update global expert count to include auxiliary experts
        self.config.num_moe_experts += self.eplb_para["num_replica"]
        # Each rank adds 1 local expert (the auxiliary one)
        self.num_local_experts += self.eplb_para["replicated_expert_per_rank"] 
        # Compute local expert indices after expansion
        expert_indices = list(range(self.config.num_moe_experts))
        start_idx =  self.ep_rank *  self.num_local_experts
        end_idx =  start_idx + self.num_local_experts
        self.local_expert_indices  = expert_indices[start_idx:end_idx]
        # Will be set later: global expert index after EPLB-aware token redistribution
        self.new_global_expert_indices = None
        # Optional: store actual token workload per expert for analysis
        self.workload_distribution = None
        print(f"RANK[{self.ep_rank}] :")
        print("\n[EPLB CONFIGURATION]")
        for k, v in self.eplb_para.items():
            print(f"  {k}: {v}")
        print("original_global_expert_indices",self.original_global_expert_indices)
        print("local_expert_indices", self.local_expert_indices  )
        print()
        
    
    def eplb(self,routing_map):
        assert os.getenv("EPLB") == "1", "EPLB is not enabled"
        # routing_map: [num_tokens, num_experts], e.g., [T, 8] for Mixtral
        # Result: workload_distribution: [1, num_original_experts]
        self.workload_distribution =   routing_map.sum(dim=0).unsqueeze(0)  
        assert self.workload_distribution.ndim == 2, "workload_distribution must be 2D"
        assert self.workload_distribution.shape[0] == 1, "The first dimension of workload_distribution must be 1"
        # Rebalance token-to-expert load across expert groups
        phy2log, log2phy, _ = rebalance_experts(self.workload_distribution ,  
                                                self.config.num_moe_experts , 
                                                self.eplb_para["num_groups"], 
                                                self.eplb_para["num_nodes"], 
                                                self.eplb_para["num_gpus"])
        # Store new expert placement 
        self.new_global_expert_indices = phy2log.flatten().tolist()
        print("After EPLB, new_global_expert_indices: ", self.new_global_expert_indices)
        # Remap weight values based on the new expert assignment
        self.eplb_modify_weights()
    def eplb_modify_weights(self):
        assert self.new_global_expert_indices != None
        assert self.auxiliary_cpu_experts_weight != None
        # Determine current rank's expert slice in the global expert mapping
        start_idx =  self.ep_rank *  self.num_local_experts
        end_idx =  start_idx + self.num_local_experts
        global_expert_indices = self.new_global_expert_indices [start_idx:end_idx]
        for i in range(len(global_expert_indices)):
            global_id = global_expert_indices [i]
            local_id = i
            # ------- fc1 weight ----
            fc1_global = self.auxiliary_cpu_experts_weight[f"linear_fc1.weight{global_id}"]
            fc1_local = getattr(self.experts.linear_fc1, f"weight{local_id}")
            assert fc1_global.shape == fc1_local.shape, f"Shape mismatch in fc1: {fc1_global.shape} vs {fc1_local.shape}"
            fc1_global = fc1_global.to(fc1_local.device)
            with torch.no_grad():
                fc1_local.data.copy_(fc1_global)

            # ------- fc2 weight -------
            fc2_global = self.auxiliary_cpu_experts_weight[f"linear_fc2.weight{global_id}"]
            fc2_local = getattr(self.experts.linear_fc2, f"weight{local_id}")
            assert fc2_global.shape == fc2_local.shape, f"Shape mismatch in fc2: {fc2_global.shape} vs {fc2_local.shape}"
            fc2_global = fc2_global.to(fc2_local.device)
            with torch.no_grad():
                fc2_local.data.copy_(fc2_global)
    def replicate_modify_weights(self):
        if self.replicate_weights_ready:
            print(f"[RANK {self.ep_rank}] Already Update New expert weight")
            return 
        self.replicate_weights_ready = True
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
                self.replicate_modify_weights()
            if int(os.getenv("EPLB", "0")) == 1:
                if self.auxiliary_cpu_experts_weight == None:
                # Preload all experts wight to CPU, to be used in reorganizeing experts
                    path = os.getenv("EXPERT_PATH")
                    print("self.layer_number",self.layer_number)
                    self.auxiliary_cpu_experts_weight = load_expert_cpu(path,self.layer_number)
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
            if int(os.getenv("EPLB", "0")) == 1:
                self.eplb(routing_map)
                # update probs, routing_map
                routing_map = eplb_modify(self.original_global_expert_indices,self.new_global_expert_indices, routing_map)
                probs = eplb_modify(self.original_global_expert_indices,self.new_global_expert_indices, probs)
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
        #     torch.save(output, f"/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/REPLICATE/{get_rank()}.pt")
        # if int(os.getenv("REPLICATE", "0")) == 1:
        #     torch.save(output, f"/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/REPLICATE/{get_rank()}_REPLICATE.pt") 
        if int(os.getenv("EPLB", "0")) == 0:
            torch.save(output, f"/root/MG_test/mixtral/EPLB/{get_rank()}.pt")
        if int(os.getenv("EPLB", "0")) == 1:
            torch.save(output, f"/root/MG_test/mixtral/EPLB/{get_rank()}_EPLB.pt")
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

def eplb_modify( original_indices , new_indices, data) -> torch.Tensor:
    """
    Modify token-to-expert routing map after EPLB (Expert Load Balancing).

    Args:
        original_indices (List[int]): Expert IDs before EPLB reordering (length = num_experts).
        new_indices (List[int]): New expert IDs after EPLB reassignment (length = num_local_experts * num_ranks).
        data (torch.Tensor): Original routing map/probs, shape = [num_tokens, num_experts] 

    Returns:
        torch.Tensor: Adjusted routing map with shape [num_tokens, len(new_indices)],
                      ensuring no token is routed twice to same expert.
    """
    # ⚠️ Currently only supports each expert being replicated at most once
    counts = Counter(new_indices)
    # Check that no expert is assigned more than twice
    for expert_id, count in counts.items():
        if count > 2:
            raise ValueError(f"Expert {expert_id} is assigned {count} times (only support up to 2)")
    NUM_EXPERTS = data.shape[1]
    assert len(original_indices) == NUM_EXPERTS, "Mismatch in expert index count"
    counts = Counter(new_indices) # Count how many times each expert appears
    visited = [0] * NUM_EXPERTS   # Track how many times each expert is visited
    results = []
    for i in new_indices:
        col = data[:, i] # Original column for expert i
        col_new = torch.zeros_like(col, dtype= data.dtype)   # Create empty column with same shape
        if counts[i] == 1:
            # Only used once: copy entire column
            col_new = col.clone()
        else:
            # Used multiple times: split tokens
            true_indices = torch.nonzero(col, as_tuple=True)[0]
            mid = len(true_indices) // 2
            if visited[i] == 0:
                chosen = true_indices[:mid] # First time: assign first half
                visited[i] += 1
            else:
                chosen = true_indices[mid:] # Second time: assign second half
            col_new[chosen] = True
        results.append(col_new.unsqueeze(1))
    final_result = torch.cat(results, dim=1) # Final shape: [num_tokens, len(new_indices)]
    return final_result
    
    
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

def load_expert_cpu(path,layer):
    """
    Load expert weights for a specific layer from a saved model checkpoint into CPU memory.

    Args:
        path (str): Path to the checkpoint file (e.g., model_optim_rng.pt).
        layer (int): 1-based layer index to load expert weights from (e.g., layer=1 for decoder.layers.0).

    Returns:
        dict: A dictionary mapping keys like "linear_fc1.weight0" to their corresponding expert tensors.
    """
    print("Preload expert weights to CPU for Layer ",layer)
    # Load full model checkpoint from disk to CPU 
    state  = torch.load(path,map_location="cpu", weights_only=False)
    model_state = state["model"]
    new_state = {}
    # Traverse all parameters and extract those belonging to experts in the specified layer
    for k, v in model_state.items():
        if "experts" in k and "_extra_state" not in k:
            # e.g. `linear_fc1.weight7` or `linear_fc2.weight7`
            parts = k.split(".")
            layer_id = int(parts[2])
            # Note: `layer` (self.layer_number) is 1-based, but the model key uses 0-based indexing
            if layer_id == layer - 1:
                new_key = ".".join(k.split("experts.")[-1].split(".")[-2:])
                new_state[new_key] = v
                # print(f"{k} --> {new_key}", v.shape,v.device)
    # print(new_state.keys())
    return  new_state