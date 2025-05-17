# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import torch.distributed as dist
from torch.distributed import get_rank

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
        print(f"[Rank {get_rank()}] Local experts: {self.num_local_experts}, Indices: {self.local_expert_indices}")
        # Initialize router
        self.router = TopKRouter(config=self.config)

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
            if int(os.getenv("DBUG", "0")) == 1:
                print(f"RANK[{rank}] : hidden_states.shape = {hidden_states.shape}")
            if int(os.getenv("IDEAL", "0")) == 1:
                idel_routing_map,_ = generate_balanced_routing_map(
                    token_num = hidden_states.shape[0]*hidden_states.shape[1],
                    num_experts = self.config.num_moe_experts ,
                    topk = self.config.moe_router_topk,
                    device =  hidden_states.device 
                )
            probs, routing_map = self.router(hidden_states)
            if int(os.getenv("SKEW", "0")) == 1:
                get_imbalanced_routing_map(
                    routing_map, expert_id=0, enforce_row_count= 1000
                )
            if int(os.getenv("MOE_TIME", "0")) == 1:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
            ##############################################################
            if int(os.getenv("IDEAL", "0")) == 1:       
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    hidden_states, probs, idel_routing_map
                )
                # if rank == 0:
                    # print(f"RANK[{rank}] : idel_routing_map",idel_routing_map.tolist())
                    # print(f"idel_routing_map.sum(dim=0) = {idel_routing_map.sum(dim=0)}")  # 每个 expert 的 token 数量
            else:
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    hidden_states, probs, routing_map
                )
                # if rank == 0:
                    # print(f"RANK[{rank}] : routing_map",routing_map.tolist())
                    # print(f"routing_map.sum(dim=0) = {routing_map.sum(dim=0)}")  # 每个 expert 的 token 数量
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)
            #####################################################
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