# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import time
from megatron.predictor.global_predictor_controller import   get_predictor_controller
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union,Dict,List
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
import requests
import struct
from megatron.predictor.global_predictor_controller import set_predictor_controller,get_predictor_controller
from concurrent.futures import ThreadPoolExecutor
import ast
from megatron.core.transformer.moe.tensor_utils_pybind import IPCHandleManager, tensor_restore_from_handler_pybind,merge_tensors_and_export_ipc_handle
import socket
from megatron.core.transformer.moe.util_transmission import pack_metadata, unpack_metadata, dtype_to_code, code_to_dtype
# import aiohttp
# import asyncio
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
        if int(os.getenv("Async_Predict", "0")) == 1 and os.getenv("EXTERNAL_EXPERTS") == "1":
            self.controller = None
        if os.getenv("EXTERNAL_EXPERTS") == "1":
            self.container_executor = ThreadPoolExecutor(max_workers=1)

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
    def _call_container(self, url, handle, metadata_list, layer_id):
        T2=time.time()
        sock_path = "/home/ubuntu/Codespace/serverless-moe/sock/socket.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)

        # Header: total metadata count (4 bytes)
        header = struct.pack('I', len(metadata_list))
        
        # Payload:
        payload = header
        payload += handle
        payload += struct.pack('i', layer_id)
        for md in metadata_list:
            payload += pack_metadata(md)

        s.sendall(payload)

        # Receive: return handle + int + metadata
        returned = s.recv(64 + 8 + 32)
        handle_bytes = returned[:64]
        result_double = struct.unpack('d', returned[64:72])[0]  # 8 bytes double
        packed_metadata = unpack_metadata(returned[72:104])
        T3=time.time()
        
        print(f"RANK[{dist.get_rank()}] Layer[{self.layer_number}] {'total time used in container':<25} {((T3 - T2) * 1000):>8.2f} ms")
        return handle_bytes, result_double, packed_metadata
            
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
        if dist.get_rank() == 0:
            print("======================================")
        def custom_forward(hidden_states):
            rank = dist.get_rank()
            if int(os.getenv("DEBUG", "0")) == 1:
                print(f"RANK[{rank}] Layer[{self.layer_number}] hidden_states.shape = {hidden_states.shape}")
            # if int(os.getenv("REPLICATE", "0")) == 1:
            #     self.replicate_modify_weights()
            if int(os.getenv("EPLB", "0")) == 1:
                if self.auxiliary_cpu_experts_weight == None:
                # Preload all experts wight to CPU, to be used in reorganizeing experts
                    path = os.getenv("EXPERT_PATH")
                    print("self.layer_number",self.layer_number)
                    self.auxiliary_cpu_experts_weight = load_expert_cpu(path,self.layer_number)
            probs, routing_map = self.router(hidden_states)
            print("probs",probs.shape, probs.dtype, "routing_map", routing_map.shape,routing_map.dtype )
            if int(os.getenv("IDEAL", "0")) == 1:
                # create ideal routing_map
                idel_routing_map,ideal_probs ,= generate_balanced_routing_map(
                    token_num = hidden_states.shape[0]*hidden_states.shape[1],
                    num_experts = self.config.num_moe_experts ,
                    topk = self.config.moe_router_topk,
                    device =  hidden_states.device 
                )
                routing_map = idel_routing_map
                probs = ideal_probs
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
                
            print(f"RANK[{dist.get_rank()}] Layer[{self.layer_number}] workload" , routing_map.sum(dim=0).long())
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
            # print(f"[Rank {rank}] after token_permutation")

            if os.getenv("EXTERNAL_EXPERTS") == "0":
                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            else:
                if os.getenv("Async_Predict") == "1": 
                    if self.controller == None:
                        self.controller = get_predictor_controller()
                    # every layer have different replicate plan
                    expert_replica_count_per_index = self.controller.expert_replica_count_per_index_map[self.layer_number]
                else:
                    # default expert_replica_count_per_index
                    # NOTE: This must match the expert indices initialized in this container.
                    # For example, if container experts are [0,1,2,3], the result will be:
                    # {0:1, 1:1, 2:1, 3:1}
                    # every layer are the same by default
                    CONFIG_PATH = os.environ.get("REPLICA_CONFIG")
                    with open(os.path.join(CONFIG_PATH, "replica.json"), "r") as f:
                        config = json.load(f)
                    rank = dist.get_rank()
                    rank_config = config["Container"][f"RANK{rank}"]
                    EXPERTS = ast.literal_eval(rank_config["EXPERTS"])
                    num_experts = len(EXPERTS[self.layer_number-1])
                    expert_replica_count_per_index = {i:1 for i in range (num_experts)}
                
                print(f"RANK[{dist.get_rank()}] Layer[{self.layer_number}] expert_replica_count_per_index" ,expert_replica_count_per_index)
                # If this rank has no assigned replicas, it will skip container execution
                # (i.e., this rank does not need to launch expert computation)
                if not expert_replica_count_per_index:
                    print(f"RANK[{dist.get_rank()}] Layer[{self.layer_number}]: no replica")
                    expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
                else:
                    start_split = time.time()
                    
                    
                    print(f"RANK[{dist.get_rank()}] Layer[{self.layer_number}]:" ,expert_replica_count_per_index)
                    local_dispatched_input, local_tokens_per_expert, local_expert_indices,\
                    container_dispatched_input, container_tokens_per_expert, container_expert_indices=split_dispatched_for_replicated_experts(
                    dispatched_input,
                    tokens_per_expert,
                    expert_replica_count_per_index
                    )
                    end_split = time.time()
                    print(f"RANK[{rank}] Layer[{self.layer_number}] Split dispatched_input: {(end_split - start_split) * 1000:.2f} ms")
                    # TODO:  Serverful expert and container expert are currently executed sequentially.
                    # This should be changed to run in parallel or asynchronously to improve performance.
                    if self.ep_rank == 0:
                        url = f"http://{os.getenv('EXPERTS_ADDRESS', '')}:5001/forward"
                    elif self.ep_rank == 1:
                        url = f"http://{os.getenv('EXPERTS_ADDRESS', '')}:5002/forward"
                    else:
                        raise ValueError(f"Unsupported expert parallel rank: {self.ep_rank}. Current test only supports ranks 0 and 1.")
                    # print(f"url:{url}")
                    handler, metadata_list = get_handler_and_tensor_metadata([container_dispatched_input,container_tokens_per_expert])
                    start_container = time.time()
                    # ==== 1. Launch asynchronous container call  ====
                    T0 = time.time()
                    container_future = self.container_executor.submit(
                            self._call_container,
                            url= url,
                            handle=handler,
                            metadata_list=metadata_list,
                            layer_id=self.layer_number - 1
                        )
                    T1 = time.time()
                    end_container = time.time()
                    print(f"RANK[{rank}] Layer[{self.layer_number}] Call Container: {(end_container - start_container) * 1000:.2f} ms")
                    # ==== 2. Run local expert computation in parallel  ====
                    local_output, mlp_bias = self.experts(local_dispatched_input,local_tokens_per_expert )
                    T2 = time.time()
                    # ==== 3. Wait for container result (blocking happens here)  ====
                    handler, latency_ms, metadata  = container_future.result()  # blocks only here
                    T3 = time.time()
                    
                    # 4. 剩余部分是文件内容
                    # handler = raw_data[4 + meta_length :]
                    print(f"returned meta:{metadata}")
                    metadata['dtype'] = str(metadata['dtype'])
                    
                    handle_manager = IPCHandleManager(handler, metadata['device'])
                    container_output = tensor_restore_from_handler_pybind(handle_manager, metadata)
                    # print(f"response output_tensor:{container_output}")
                    print(f'RANK[{rank}] Layer[{self.layer_number}] Forward time in Container: {latency_ms} ms')
                    # print(f"RANK[{rank}] Layer[{self.layer_number}] {'Async request time:':<25} {((T1 - T0) * 1000):>8.2f} ms")
                    # print(f"RANK[{rank}] Layer[{self.layer_number}] {'Local forward time:':<25} {((T2 - T1) * 1000):>8.2f} ms")
                    # ==== 4. Combine local and container expert outputs ====
                    start_combine = time.time()
                    expert_output =  combine_output(
                        local_output,local_tokens_per_expert,local_expert_indices,
                        container_output, container_tokens_per_expert, container_expert_indices
                    )
                    end_combine= time.time()
                    print(f"RANK[{rank}] Layer[{self.layer_number}] Combine dispatched_input: {(end_combine - start_combine) * 1000:.2f} ms")
                    # handle_manager.close_ipc_handle()
            # ==== 5. Restore original token order ====
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
                print(f"RANK[{rank}] Layer[{self.layer_number}] moe layer elapsed {elapsed} ms\n",)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)
        return output, mlp_bias

def generate_balanced_routing_map(token_num, num_experts, topk,device):
    assert topk <= num_experts, "topk must be ≤ num_experts"

    routing_map = torch.zeros((token_num, num_experts), dtype=torch.bool)
    routing_prob = torch.zeros((token_num, num_experts), dtype=torch.float32)
    expert_counts = np.zeros(num_experts, dtype=int)

    for i in range(token_num):
        # 从当前计数最小的 experts 中选择 topk 个
        topk_experts = np.argsort(expert_counts)[:topk]
        routing_map[i, topk_experts] = True
        expert_counts[topk_experts] += 1
        routing_prob[i, topk_experts] = 1.0 / topk
    routing_map = routing_map.to(device)
    routing_prob = routing_prob.to(device)
    return routing_map, routing_prob,expert_counts
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


def split_tokens_and_inputs(dispatched_input, tokens_per_expert, world_size):
    # Step 1: Split tokens_per_expert into trunks
    trunk_tokens = []
    print( f"[Rank {get_rank()}]","dispatched_input",dispatched_input.shape, "tokens_per_expert",tokens_per_expert )
    for _ in range(world_size):
        trunk_tokens.append(torch.zeros_like(tokens_per_expert))
    
    for expert_idx, num_tokens in enumerate(trunk_tokens):
        if expert_idx<len(trunk_tokens)-1:
            trunk_tokens[expert_idx].copy_(tokens_per_expert//world_size) 
        else:
            trunk_tokens[expert_idx].copy_(torch.ceil(tokens_per_expert / world_size)) 
    print( f"[Rank {get_rank()}]","trunk_tokens",trunk_tokens,)
    
    # Step 2: Split dispatched_input into trunks
    trunk_inputs = [[] for _ in range(world_size)]
    current_idx = 0
    
    for expert_idx, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            continue   
        for trunk_idx in range(world_size):
            assign = trunk_tokens[trunk_idx][expert_idx]
            trunk_inputs[trunk_idx].append(dispatched_input[current_idx:current_idx+assign])
            print(f"[Rank {get_rank()}] expert_idx {expert_idx} trunk_idx {trunk_idx}",trunk_inputs[trunk_idx][-1].shape ,"\n")
            current_idx += assign
    
    # Concatenate each trunk's inputs
    trunk_inputs = [torch.cat(inputs) for inputs in trunk_inputs]
    
    return trunk_tokens, trunk_inputs

def split_dispatched_for_replicated_experts(
    dispatched_input: torch.Tensor,               
    tokens_per_expert: torch.Tensor,              
    expert_replica_count_per_index: Dict[int, int]     
):
    '''
    dispatched_input [T, D]: The original input tokens assigned to all experts.
    tokens_per_expert: Number of tokens assigned to each expert
    expert_replica_count_per_index: {expert_id: num_container_replica}: Which experts are replicated and how many container replicas they have.
    
    '''
    D = dispatched_input.shape[1]
    E = tokens_per_expert.shape[0]
    # Compute starting index for each expert's tokens in the flattened dispatched_input
    start_indices = torch.cat([
        torch.zeros(1, dtype=torch.long),
        tokens_per_expert.cumsum(dim=0)[:-1]
    ])  # shape: [E]
    # print(start_indices)
    
    # Local expert input parts
    local_parts = []
    local_tokens_per_expert = torch.zeros(E, dtype=tokens_per_expert.dtype, device=tokens_per_expert.device)
    local_expert_indices = list(range(E))
    
    # Container (replicated) expert input parts
    container_parts = []
    container_tokens_per_expert = []
    container_expert_indices = []
    
    for expert_id in range(E):
        num_tokens = tokens_per_expert[expert_id].item()
        if num_tokens == 0:
            local_tokens_per_expert[expert_id] = 0
            if expert_id in expert_replica_count_per_index:
                container_expert_indices += [expert_id]
                container_tokens_per_expert += [0]
            continue
        # Slice input for current expert
        start = start_indices[expert_id].item()
        end = start + num_tokens
        expert_input = dispatched_input[start:end]  # shape [num_tokens, D]
        if expert_id in expert_replica_count_per_index:
            num_container_expert = expert_replica_count_per_index[expert_id]
            total_replica = 1 + num_container_expert  # local + container replicas
            # Evenly split input tokens among all replicas
            chunks = torch.tensor_split(expert_input, total_replica, dim=0)
            # First chunk goes to the local expert
            local_parts.append(chunks[0])                     
            local_tokens_per_expert[expert_id] = chunks[0].shape[0]
            # Remaining chunks go to container replicas
            for chunk in chunks[1:]:
                container_parts.append(chunk)              
            container_expert_indices += ([expert_id] * num_container_expert)
            container_tokens_per_expert += [ chunk.shape[0] for chunk in chunks[1:]]
        else:
            # Expert is not replicated
            local_parts.append(expert_input)
            local_tokens_per_expert[expert_id] = expert_input.shape[0]  
    local_dispatched_input = torch.cat(local_parts, dim=0) if local_parts else torch.empty((0, D))
    container_dispatched_input = torch.cat(container_parts, dim=0) if container_parts else torch.empty((0, D))
    container_tokens_per_expert = torch.tensor(container_tokens_per_expert, dtype= tokens_per_expert.dtype, device=tokens_per_expert.device)
    assert local_tokens_per_expert.sum() == local_dispatched_input.shape[0], \
        f"Mismatch: local token sum {local_tokens_per_expert.sum().item()} != local dispatched_input size {local_dispatched_input.shape[0]}"
    assert container_tokens_per_expert.sum() == container_dispatched_input.shape[0], \
        f"Mismatch: container token sum {container_tokens_per_expert.sum().item()} != container dispatched_input size {container_dispatched_input.shape[0]}"
    assert local_dispatched_input.shape[0] + container_dispatched_input.shape[0] == dispatched_input.shape[0], \
        "Mismatch: combined token count does not equal total dispatched_input token count"
    assert sum(expert_replica_count_per_index.values()) == len(container_expert_indices), \
        f"Mismatch: number of container replicas {expert_replica_count_per_index.values()} != recorded container_expert_indices length {len(container_expert_indices)}"

    return (local_dispatched_input, 
            local_tokens_per_expert, 
            local_expert_indices,
            
            container_dispatched_input.to("cuda"), 
            container_tokens_per_expert.to("cuda"),
            container_expert_indices)

def combine_output(
    local_output: torch.Tensor,
    local_tokens_per_expert: torch.Tensor,
    local_expert_indices: List[int],
    container_output: torch.Tensor,
    container_tokens_per_expert: torch.Tensor,  
    container_expert_indices: torch.Tensor,
):  
    """
    Construct the output by combining the outputs from local experts
    and container (replicated) experts.
    Args:
        local_output (Tensor): [T1, D] The output computed by local experts.
        local_tokens_per_expert (Tensor): [E], number of tokens handled by each local expert.
        local_expert_indices (List[int]): Indices of local experts (typically range(E)).
        
        container_output (Tensor): [T2, D] The output computed by container replicas.
        container_tokens_per_expert (Tensor): [R], number of tokens handled by each container replica.
        container_expert_indices (Tensor): [R], expert index for each container replica out
    """
    D = local_output.shape[1]
    E = len(local_expert_indices)
    
    # Initialize expert output buckets
    expert_outputs = [[] for _ in range(E)]
    # Assign slices of local_output to corresponding experts
    local_offset = 0
    for expert_id in range(E):
        num_tokens = local_tokens_per_expert[expert_id].item()
        if num_tokens > 0:
            local_expert_output = local_output[local_offset: local_offset + num_tokens]
            expert_outputs[expert_id].append(local_expert_output)
            local_offset += num_tokens
    # Assign slices of container_output to corresponding replicated experts
    container_offset = 0
    for i, expert_id in enumerate(container_expert_indices):
        num_tokens = container_tokens_per_expert[i].item()
        if num_tokens > 0:
            container_expert_output = container_output[container_offset: container_offset + num_tokens]
            expert_outputs[expert_id].append(container_expert_output)
            container_offset += num_tokens
    # Concatenate all expert outputs in expert ID order to recover original token layout
    full_output = torch.cat([
        torch.cat(parts, dim=0) if parts else torch.empty((0, D), device=local_output.device)
        for parts in expert_outputs
    ], dim=0)
    return full_output


def get_dtype_size(dtype):
    """获取dtype的字节大小"""
    return torch.tensor([], dtype=dtype).element_size()
    

def get_handler_and_tensor_metadata(tensors):

    handler = merge_tensors_and_export_ipc_handle(tensors, tensors[0].device.index)
    # completed preparing handler

    max_dtype = max(tensors, key=lambda t: get_dtype_size(t.dtype)).dtype
    max_dtype_size = get_dtype_size(max_dtype)
    total_elements = 0
    metadata = []
    offset_bytes = 0
    for tensor in tensors:
        # 计算当前张量需要的元素数（考虑对齐）
        tensor_bytes = tensor.numel() * get_dtype_size(tensor.dtype)
        elements_needed = (tensor_bytes + max_dtype_size - 1) // max_dtype_size
        shape = list(tensor.shape) + [0] * (3 - len(tensor.shape))
        # 记录元数据
        metadata.append({
            'dtype': tensor.dtype,
            'shape': tensor.shape,
            'device': tensor.device.index,
            'offset_bytes':offset_bytes
        })
        offset_bytes += tensor_bytes
        total_elements += elements_needed
    return handler, metadata