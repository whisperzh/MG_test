from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
import pickle
import os
import torch
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

DEBUG= False
RANK=0
PATH="/root/MG_test/mixtral/REPLICATE/saved_objects"
RANK_PATH=f"rank_{RANK}"
NUM_LOCAL_EXPERTS=4
os.environ["IN_CONTAINER"]="1"
GPU_IDX = 0
cuda_device = f"cuda:{GPU_IDX}"

def init_experts():
    source_dir=os.path.join(PATH,RANK_PATH)
    
    source_file_submodules_experts = os.path.join(source_dir, "submodules_experts.pickle")
    source_file_config = os.path.join(source_dir, "config.pickle")
    
    with open(source_file_submodules_experts, 'rb') as f:
        submodules_experts = pickle.load(f)

    with open(source_file_config, 'rb') as f:
        config = pickle.load(f)
        
    if DEBUG:
        print(config)
        return None
    
    experts = build_module(submodules_experts, NUM_LOCAL_EXPERTS, config)
    return experts

# experts=init_experts()

# pure for test purpose
def _test():
    tokens_per_expert=torch.tensor([2,0,2,2],dtype=torch.int32).to("cuda")
    dispatched_input=torch.randn(6,4096,dtype=torch.bfloat16).to("cuda")
    output, mlp_bias = experts(dispatched_input, tokens_per_expert)
    print(output)
    print(mlp_bias)
    print("experts finished calculating")
    

@app.route("/forward", methods=["POST"])
def forward():
    dispatched_input = request.json["dispatched_input"]
    tokens_per_expert = request.json["tokens_per_expert"]
    
    dispatched_input = torch.tensor(request.json["dispatched_input"],dtype=torch.bfloat16).to(cuda_device)
    tokens_per_expert = torch.tensor(request.json["tokens_per_expert"],dtype=torch.int32).to(cuda_device)
   
    torch.cuda.set_device(GPU_IDX)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    output, mlp_bias = experts(dispatched_input, tokens_per_expert)
    print(f"mlp_bias: {mlp_bias}")

    end_event.record()
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event)
    
    return jsonify({"hidden_output": output.cpu().tolist(),"latency_ms":latency_ms})


if __name__=="__main__":
    
    # app.run(host="0.0.0.0", port=5000)

    a = torch.load("/root/MG_test/mixtral/mixtral-mcore-TP1PP1EP2Layer1/iter_0000001/mp_rank_00_000/model_optim_rng.pt",weights_only=False)
    lis=a['model'].keys()
    for k in lis:
        print(k)