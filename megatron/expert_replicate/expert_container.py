
import pickle
import os
import torch

from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from flask import Flask, request, jsonify
from megatron.training.checkpointing import _load_base_checkpoint

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
from megatron.core.transformer.spec_utils import ModuleSpec
import json

app = Flask(__name__)

DEBUG = bool(os.environ.get("DEBUG", False))
RANK = int(os.environ.get("RANK", 0))
PATH_SAVEDOBJ = os.environ.get("PATH_SAVEDOBJ", "/root/MG_test/mixtral/REPLICATE/saved_objects")
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 3))
GPU_IDX = int(os.environ.get("GPU_IDX", 0))
WARMUP = bool(os.environ.get("WARMUP", True))
LAYER = int(os.environ.get("LAYER", 0))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/root/MG_test/mixtral/mixtral-mcore-TP1PP1EP2Layer1")


with open(f"{PATH_SAVEDOBJ}/rank_{RANK}/args.pickle", 'rb') as f:
    args = pickle.load(f)
##

rank0 = True
checkpointing_context = None

RANK_PATH = f"rank_{RANK}"
cuda_device = f"cuda:{GPU_IDX}"

os.environ["IN_CONTAINER"] = "1"


def init_experts():
    source_dir = os.path.join(PATH_SAVEDOBJ, RANK_PATH)

    expert_module = TEGroupedMLP
    expert_submodule = MLPSubmodules(
        linear_fc1=TEColumnParallelGroupedLinear,
        linear_fc2=TERowParallelGroupedLinear)

    submodules_experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # source_file_submodules_experts = os.path.join(source_dir, "submodules_experts.pickle")

    # with open(source_file_submodules_experts, 'rb') as f:
    #     submodules_experts = pickle.load(f)

    source_file_config = os.path.join(source_dir, "config.pickle")
    with open(source_file_config, 'rb') as f:
        config = pickle.load(f)

    if DEBUG:
        print(config)
        return None

    experts = build_module(submodules_experts, NUM_LOCAL_EXPERTS, config)
    load_expert_weights(experts, WEIGHT_PATH, [0, 1, 2], strict=False)
    return experts


def load_expert_weights(
    expert_model,
    checkpoint_path,
    expert_indices,  # 例如 [0,2,4,5]
    strict=True,
    checkpoint_format="torch",
):
    """
    Load specific expert weights from checkpoint.
    """

    # 加载检查点
    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        checkpoint_path,
        args=args,  # 这里需要你的 args
        rank0=True,
        sharded_state_dict=None,
    )

    if state_dict is None:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # 获取模型状态字典
    model_state_dict = state_dict['model']

    # name in model     linear_fc1.weight0
    # real weight name  decoder.layers.0.mlp.experts.linear_fc1.weight0

    for expert_idx in expert_indices:
        # 构建expert相关的键名
        expert_keys = [f'decoder.layers.{LAYER}.mlp.experts.linear_fc1.weight{expert_idx}',
                       f'decoder.layers.{LAYER}.mlp.experts.linear_fc2.weight{expert_idx}']

        # 提取当前expert的权重
        expert_state_dict = {
            f"linear_fc1.weight{expert_idx}": v for k, v in model_state_dict.items()
            if any(k.startswith(prefix) for prefix in expert_keys)
        }
        print(f"expert_state_dict keys: {expert_state_dict.keys()}")

        if not expert_state_dict:
            print(f"Warning: No weights found for expert {expert_idx}")
            continue

        # 加载到对应的expert
        try:
            with torch.no_grad():
                actual_expert_weight_fc1 = getattr(expert_model.linear_fc1, f"weight{expert_idx}")
                actual_expert_weight_fc1.data.copy_(
                    model_state_dict[f'decoder.layers.{LAYER}.mlp.experts.linear_fc1.weight{expert_idx}'].data)

                actual_expert_weight_fc2 = getattr(expert_model.linear_fc2, f"weight{expert_idx}")
                actual_expert_weight_fc2.data.copy_(
                    model_state_dict[f'decoder.layers.{LAYER}.mlp.experts.linear_fc2.weight{expert_idx}'].data)
        except Exception as e:
            print(f"Error loading expert {expert_idx}: {str(e)}")

    return expert_model


experts = init_experts()
print(experts.linear_fc1.weight0)

# pure for test purpose


def _warmup():
    # warmup with 32 * NUM_LOCAL_EXPERTS tokens per expert
    tokens_per_expert = torch.tensor([32]*NUM_LOCAL_EXPERTS, dtype=torch.int32).to(cuda_device)
    dispatched_input = torch.randn(1, 32*NUM_LOCAL_EXPERTS, 4096,
                                   dtype=torch.bfloat16).to(cuda_device)
    output, mlp_bias = experts(dispatched_input, tokens_per_expert)
    print("experts finished warmup")

@app.route("/forward", methods=["POST"])
def forward():
    dispatched_input = request.json["dispatched_input"]
    tokens_per_expert = request.json["tokens_per_expert"]

    dispatched_input = torch.tensor(
        request.json["dispatched_input"], dtype=torch.bfloat16).to(cuda_device)
    tokens_per_expert = torch.tensor(
        request.json["tokens_per_expert"], dtype=torch.int32).to(cuda_device)

    torch.cuda.set_device(GPU_IDX)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    output, mlp_bias = experts(dispatched_input, tokens_per_expert)
    print(f"mlp_bias: {mlp_bias}")

    end_event.record()
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event)

    return jsonify({"hidden_output": output.cpu().tolist(), "latency_ms": latency_ms})

if __name__ == "__main__":
    if WARMUP and not DEBUG:
        _warmup()
    app.run(host="0.0.0.0", port=5000)
