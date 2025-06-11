
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
EXPERTS = list(os.environ.get("EXPERTS", [[0, 1, 2],[0, 1,2,3]]))

GPU_IDX = int(os.environ.get("GPU_IDX", 0))
WARMUP = bool(os.environ.get("WARMUP", True))
LAYER = list(os.environ.get("LAYER", [0, 1]))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/root/MG_test/weights")
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

    source_file_config = os.path.join(source_dir, "config.pickle")
    with open(source_file_config, 'rb') as f:
        config = pickle.load(f)

    if DEBUG:
        print(config)
        return None

    experts = [build_module(submodules_experts, len(e), config) for e in EXPERTS]
    for idx, e in enumerate(experts):
        load_expert_weights(e, EXPERTS[idx], LAYER[idx])
    return experts


def load_expert_weights(
    expert_model,
    expert_indices,  # 例如 [0,2,4,5]
    layer
):
    """
    Load specific expert weights from checkpoint.
    """

    for expert_idx in expert_indices:
        try:
            with torch.no_grad():
                target_prefix_fc1 = f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert_idx}"
                target_prefix_fc2 = f"decoder.layers.{layer}.mlp.experts.linear_fc2.weight{expert_idx}"

                path_fc1 = os.path.join(WEIGHT_PATH, target_prefix_fc1.replace(".", "_")+".pt")
                path_fc2 = os.path.join(WEIGHT_PATH, target_prefix_fc2.replace(".", "_")+".pt")

                weight_fc1 = torch.load(path_fc1, weights_only=True).to(cuda_device)
                weight_fc2 = torch.load(path_fc2, weights_only=True).to(cuda_device)
                
                actual_expert_weight_fc1 = getattr(
                    expert_model.linear_fc1, f"weight{expert_idx}")
                actual_expert_weight_fc1.data.copy_(weight_fc1)

                actual_expert_weight_fc2 = getattr(
                    expert_model.linear_fc2, f"weight{expert_idx}")
                actual_expert_weight_fc2.data.copy_(weight_fc2)
                    
        except Exception as e:
            print(f"Error loading expert {expert_idx}: {str(e)}")

    return expert_model


experts = init_experts()
print(experts[0].linear_fc1.weight0)

# pure for test purpose

def _warmup():
    # warmup with 32 * NUM_LOCAL_EXPERTS tokens per expert

    for idx,e in enumerate(experts):
        tokens_per_expert = torch.tensor([32]*len(EXPERTS[idx]), dtype=torch.int32).to(cuda_device)
        dispatched_input = torch.randn(1, 32*len(EXPERTS[idx]), 4096,
                                   dtype=torch.bfloat16).to(cuda_device)
        output, mlp_bias = e(dispatched_input, tokens_per_expert)
    print("experts finished warmup")


@app.route("/forward", methods=["POST"])
def forward():
    torch.cuda.set_device(GPU_IDX)
    
    dispatched_input = request.json["dispatched_input"]
    tokens_per_expert = request.json["tokens_per_expert"]
    target_layer = int(request.json["layer"]) if request.json["layer"]!=None else 0

    dispatched_input = torch.tensor(
        request.json["dispatched_input"], dtype=torch.bfloat16).to(cuda_device)
    tokens_per_expert = torch.tensor(
        request.json["tokens_per_expert"], dtype=torch.int32).to(cuda_device)


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()


    output, mlp_bias = experts[target_layer](dispatched_input, tokens_per_expert)
    print(f"mlp_bias: {mlp_bias}")

    end_event.record()
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event)

    return jsonify({"hidden_output": output.cpu().tolist(), "latency_ms": latency_ms})


if __name__ == "__main__":
    # assert False, "we got a big problem, different layer should be separated "
    if WARMUP and not DEBUG:
        _warmup()
    app.run(host="0.0.0.0", port=5000)
