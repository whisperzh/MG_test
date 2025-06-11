import torch
import os
import argparse
import re
from collections import OrderedDict
from transformers import AutoModelForCausalLM
from transformers import MixtralConfig
def get_param_name(model, target_param):
    for name, param in model.named_parameters():
        if param is target_param:
            return name
    return None  # 如果找不到，返回 None

parser = argparse.ArgumentParser(description='Modify model checkpoint file paths.')
parser.add_argument('--root_dir', type=str, help='Root directory containing model files', default="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/mixtral-mcore-TP1PP1EP4Layer1")
parser.add_argument('--hf_path', type=str,  help='Root directory containing model files', default="/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1")
# this is only for mixtral
args = parser.parse_args()
root_dir = args.root_dir
hf_path =  args.hf_path
config =  MixtralConfig.from_pretrained(hf_path)
hf_model = AutoModelForCausalLM.from_pretrained(hf_path , device_map="cpu")


root_dir = args.root_dir
target_name = 'model_optim_rng.pt'
if not os.path.exists(root_dir):
    print(f"not exist: {root_dir}")
    exit(0)
all_paths = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    if target_name in filenames:
        full_path = os.path.join(dirpath, target_name)
        all_paths.append(full_path)
for path in all_paths:
    print(path)
def modify_keys(path):
    state  = torch.load(path,map_location="cpu", weights_only=False)
    print(state.keys())
    print(state["model"].keys())
    model_state = state["model"]
    new_model_state = OrderedDict()
    for key, value in   model_state.items():
        if value == None:
            continue
        print(f"{key}: {value.shape}")
        if "experts.local_experts." in key  :
        # "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight", 
        # "decoder.layers.0.mlp.experts.linear_fc1.weight0", 
            if  "_extra_state" not in key:
                parts = key.split(".")
                layer_idx = parts[2]
                expert_idx = parts[6]
                fc_name = parts[7]  # linear_fc1 or linear_fc2
                param_type = parts[-1]  # weight or _extra_state     
                # 构造新 key
                if param_type == "weight":
                    new_key = f"decoder.layers.{layer_idx}.mlp.experts.{fc_name}.weight{expert_idx}"
                new_model_state[new_key] = value  
            else: 
                parts = key.split(".")
                layer_idx = parts[2]
                expert_idx = parts[6]
                fc_name = parts[7]  # linear_fc1 or linear_fc2
                param_type = parts[-1]  # weight or _extra_state     
                new_key = f"decoder.layers.{layer_idx}.mlp.experts.{fc_name}._extra_state"
                new_model_state[new_key] = value  
            print("before:")
            print(key)
            print("after")
            print(new_key)
        else:
            new_model_state[key] = value  
    state["model"] = new_model_state
    torch.save(state,path)
    return new_model_state

def check_hf_weight(path):
    match = re.search(r'TP(\d+)PP(\d+)EP(\d+)', path)
    if match:
        tp = int(match.group(1))
        pp = int(match.group(2))
        ep = int(match.group(3))
        print(f"TP: {tp}, PP: {pp}, EP: {ep}")
    else:
        print("Format not matched.")
    assert tp == 1 and pp == 1, f"Assertion failed: tp={tp}, pp={pp}. Only EP is enabled."
    match = re.search(r'mp_rank_\d+_(\d+)', path)
    if match:
        ep_rank = int(match.group(1)) 
        print(f"ep_rank: {ep_rank}")
    else:
        print("No match found.")
  
    state =  torch.load(path,map_location="cpu", weights_only=False)
    mg_model_state = state["model"]
    world_size = ep  # 总共有几个 expert ranks / GPUs
    expert_indices = list(range(config.num_local_experts))  # 所有 expert 的 index
    # 每个 rank 应该负责的 expert 数量
    num_local_experts = int(config.num_local_experts // world_size)
    # 当前 rank 负责的 expert index 范围
    start_idx = ep_rank * num_local_experts
    end_idx = start_idx + num_local_experts
    # 当前 rank 实际负责的 expert indices
    local_expert_indices = expert_indices[start_idx:end_idx]
    # 打印调试信息
    print(f"Total experts: {config.num_local_experts}")
    print(f"World size (EP): {world_size}")
    print(f"EP rank: {ep_rank}")
    print(f"Experts per rank: {num_local_experts}")
    print(f"Local expert indices: {local_expert_indices}")
    print("====================HF weight=====================")
    for name, param in hf_model.named_parameters():
        print(f"{name}: {param.shape}")
    print("====================MG weight=====================")
    for key, value in mg_model_state.items():
        print(key, value.shape)
        if "_extra_state" in key:
            continue
        if "embedding" in key:
            print(torch.equal(mg_model_state[key], hf_model.model.embed_tokens.weight   ))
        if "final_layernorm" in key:
            print(torch.equal(mg_model_state[key], hf_model.model.norm.weight   ))
        if "output_layer"  in key:
            print(torch.equal(mg_model_state[key], hf_model.lm_head.weight  ))
        if "layers" in key:
            parts = key.split(".")
            layer_idx = int(parts[2])
            hf_layer = hf_model.model.layers[ layer_idx ]
            if "self_attention" in key  :
                hf_attn = hf_layer.self_attn
                if "linear_proj" in key:
                    print(torch.equal(mg_model_state[key], hf_attn.o_proj.weight    ))
                if "linear_qkv" in key:
                    if "linear_qkv.weight" in key:
                        num_query_groups=8
                        dim = 128
                        num_querys_per_group = 4
                        num_heads = 32
                        attn_weight = torch.cat([
                        hf_attn.q_proj.weight.reshape((num_query_groups, num_querys_per_group*dim, -1)),
                        hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
                        hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
                        ], dim=1).reshape((-1, config.hidden_size))
                        print(torch.equal(mg_model_state[key],attn_weight  ))
                    if "linear_qkv.layer_norm_weight" in key:
                        print(torch.equal(mg_model_state[key], hf_layer.input_layernorm.weight    ))
            if "mlp" in key:
                hf_mlp = hf_layer.block_sparse_moe
                if "pre_mlp_layernorm" in key:
                    print(torch.equal(mg_model_state[key], hf_layer.post_attention_layernorm.weight    ))
                if "router" in  key:
                    print(torch.equal(mg_model_state[key], hf_mlp.gate.weight   ))
                if "experts.linear_fc" in  key:
                    hf_experts = hf_mlp.experts
                    match = re.search(r'weight(\d+)$', key)
                    expert_idx = int(match.group(1))
                    global_id = local_expert_indices [expert_idx]
                    if "linear_fc1" in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight =  torch.cat([
                                hf_experts[global_id].w1.weight,
                                hf_experts[global_id].w3.weight
                            ], dim=0)
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key}, {get_param_name(hf_model,  hf_experts[global_id].w1.weight) }, {get_param_name(hf_model,  hf_experts[global_id].w3.weight) }")
                            print(f"    Megatron weight shape: {mg_weight.shape}")
                            print(f"    HF weight shape      : {hf_weight.shape}")
                            print("     Update...")
                            mg_model_state[key] = hf_weight
                        else:
                            print(f"    ✅ Match in {key}")
                    if "linear_fc2"  in  key:
                        mg_weight = mg_model_state[key]
                        hf_weight =   hf_experts[global_id].w2.weight
                        if not torch.equal(mg_weight, hf_weight):
                            print(f"    ❌ Mismatch in {key},   {get_param_name(hf_model,  hf_experts[global_id].w2.weight) }")
                            print(f"    Megatron weight shape: {mg_weight.shape}")
                            print(f"    HF weight shape      : {hf_weight.shape}")
                            print("     Update...")
                            mg_model_state[key] = hf_weight
                        else:
                            print(f"    ✅ Match in {key}")
    print("save to...", path)                   
    torch.save(state,path)
    
for i in all_paths:
    modify_keys(i)
    check_hf_weight(i)
 