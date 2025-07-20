
import pickle
import os
import torch
import time
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from flask import Flask, request, jsonify
from megatron.training.checkpointing import _load_base_checkpoint

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
from megatron.core.transformer.spec_utils import ModuleSpec
import json
import ast
import threading
from tensor_utils_pybind import merge_tensors_and_export_ipc_handle, tensor_restore_from_handler_pybind, IPCHandleManager

# ======================== Init ========================
app = Flask(__name__)
DEBUG = bool(os.environ.get("DEBUG", False))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
PATH_SAVEDOBJ = os.environ.get("PATH_SAVEDOBJ", "/home/ubuntu/MG_test/mixtral/REPLICATE/saved_objects")
# expert indices for each layer
EXPERTS = ast.literal_eval(os.environ.get("EXPERTS", [[0, 1, 2,3]]))
GPU_IDX = int(os.environ.get("GPU_IDX", 0))
torch.cuda.set_device(GPU_IDX)
WARMUP = bool(os.environ.get("WARMUP", True))
# layer ids
LAYER = ast.literal_eval(os.environ.get("LAYER", [0]))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/home/ubuntu/MG_test/weights")
RANK_PATH = f"rank_{RANK}"
cuda_device = f"cuda:{GPU_IDX}"
os.environ["IN_CONTAINER"] = "1"
if len(LAYER) != len(EXPERTS):
    raise ValueError("LAYER and EXPERTS length mismatch.")
# Print all variables in formatted style
print("\n=== Loaded Environment Variables ===")
print(f"{'DEBUG':<15}: {DEBUG}")
print(f"{'WARMUP':<15}: {WARMUP}")
print(f"{'RANK':<15}: {RANK}")
print(f"{'WORLD_SIZE':<15}: {WORLD_SIZE}")
print(f"{'GPU_IDX':<15}: {GPU_IDX}")
print(f"{'WARMUP':<15}: {WARMUP}")
print(f"{'LAYER':<15}: {LAYER}")
print(f"{'EXPERTS':<15}: {EXPERTS}")
print(f"{'WEIGHT_PATH':<15}: {WEIGHT_PATH}")
print(f"{'PATH_SAVEDOBJ':<15}: {PATH_SAVEDOBJ}")
print(f"{'RANK_PATH':<15}: {RANK_PATH}")
print(f"{'CUDA Device':<15}: {torch.cuda.current_device()}")
print("====================================\n")
 



def load_expert_weights(
    expert_model,
    expert_indices,  # e.g. [0,2,4,5]
    layer,
    cpu_expert_state
):
    """
    Load specific expert weights from cpu_expert_state.
    """
    print(f"     Load expert for layer {layer}", f"expert_indices is {expert_indices}")
    for idx, expert_idx in enumerate(expert_indices):
        print(f"     idx: {idx}, expert_idx:{expert_idx}")
        try:
            with torch.no_grad():
                weight_fc1 =  cpu_expert_state[f"L{layer}_E{expert_idx}_W1"]
                weight_fc2 =  cpu_expert_state[f"L{layer}_E{expert_idx}_W2"]
                # Assign weights to the model
                actual_expert_weight_fc1 = getattr(
                    expert_model.linear_fc1, f"weight{idx}")
                actual_expert_weight_fc1.data.copy_(weight_fc1)

                actual_expert_weight_fc2 = getattr(
                    expert_model.linear_fc2, f"weight{idx}")
                actual_expert_weight_fc2.data.copy_(weight_fc2)
                    
        except Exception as e:
            print(f"Error loading expert {expert_idx}: {str(e)}")
    return expert_model


class ExpertsManager:
    def __init__(self,):
        print("Create ExpertsManager")
        self.config = self.load_config()
        # Determine which global experts belong to this rank
        # eg. E=8, RANK 1 global_expert_indices = [0,1,2,3], 
        #     E=8  RANK 2 global_expert_indices = [4,5,6,7],
        num_experts = self.config.num_moe_experts
        num_local_experts = num_experts // WORLD_SIZE
        all_expert_indices = list(range( num_experts)) 
        ep_rank = RANK
        start_idx = ep_rank * num_local_experts
        end_idx = start_idx + num_local_experts
        self.global_expert_indices = all_expert_indices[start_idx:end_idx]
        # change self.expert_indices to dict: 
        # key: layer id, value: expert indices
        self.expert_indices = {layer: EXPERTS[i] for i, layer in enumerate(LAYER)}
        self.layers = LAYER
        assert set(self.layers) == set(self.expert_indices.keys()), "Mismatch between layers and expert_indices keys"

        # Preload expert weights to CPU
        self.cpu_expert_state = self.preload_expert_cpu()
        # key: layer id, value: expert model
        self.experts =  self.init_experts()
     
        print("self.layers",self.layers)
        print("self.expert_indices",self.expert_indices)
        print(self.experts[self.layers [0]].linear_fc1.weight0.shape)
        # Initialize an event object for each layer to synchronize expert weight updates with forward requests.
        # The event ensures that a forward pass does not start until the expert weights are fully updated. 
        self.update_events = {layer_id: threading.Event() for layer_id in self.layers}
        for e in self.update_events.values():
            e.set()   # 默认已完成（初始 expert 可用）
        
    def load_config(self):
        """Load transformer configuration for this rank."""
        source_dir = os.path.join(PATH_SAVEDOBJ, RANK_PATH)
        source_file_config = os.path.join(source_dir, "config.pickle")
        with open(source_file_config, 'rb') as f:
            config = pickle.load(f)
        print("Load config",source_file_config)
        return config
    def init_experts(self):
        """Instantiate and load weights into replicated experts for each layer."""
        print("Init replicated experts")
        expert_module = TEGroupedMLP
        expert_submodule = MLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear,
            linear_fc2=TERowParallelGroupedLinear)
        submodules_experts = ModuleSpec(module=expert_module, submodules=expert_submodule)
        if DEBUG:
            print(self.config)
            return None
        print("  LAYER", self.layers)
        print("  EXPERT INDICES", self.expert_indices)
        experts = {}
        for layer_id in self.layers:
            indices = self.expert_indices[layer_id]
            expert_model = build_module(submodules_experts, len(indices), self.config)
            load_expert_weights(expert_model, indices, layer_id, self.cpu_expert_state)
            experts[layer_id] = expert_model
        return experts
    
    def preload_expert_cpu(self):
        """Preload all assigned expert weights (for this rank) into CPU memory."""
        print("Preload all the expert weights for LAYERS for RANK to CPU")
        cpu_expert_state = {}
        for layer_id in LAYER:
            for global_expert_id in self.global_expert_indices:
                target_prefix_fc1 = f"decoder.layers.{layer_id}.mlp.experts.linear_fc1.weight{global_expert_id}"
                target_prefix_fc2 = f"decoder.layers.{layer_id}.mlp.experts.linear_fc2.weight{global_expert_id}"
                path_fc1 = os.path.join(WEIGHT_PATH, target_prefix_fc1.replace(".", "_")+".pt")
                path_fc2 = os.path.join(WEIGHT_PATH, target_prefix_fc2.replace(".", "_")+".pt")
                weight_fc1 = torch.load(path_fc1, weights_only=True) # CPU 
                weight_fc2 = torch.load(path_fc2, weights_only=True) # torch.bfloat16
                print("  load",f"L{layer_id}_E{global_expert_id}_W1", "from", path_fc1)
                print("  load",f"L{layer_id}_E{global_expert_id}_W2", "from", path_fc2)
                cpu_expert_state[f"L{layer_id}_E{global_expert_id}_W1"] = weight_fc1
                cpu_expert_state[f"L{layer_id}_E{global_expert_id}_W2"] = weight_fc2
        return cpu_expert_state
    
    def _warmup(self):
        # warmup with 128 * NUM_LOCAL_EXPERTS tokens per expert
        print("Warm up ...")
        N = 128
        for _ in range(10):
            for layer_id in self.layers:
                model = self.experts[layer_id]
                indices = self.expert_indices[layer_id]
                tokens_per_expert = torch.tensor([N]*len(indices), dtype=torch.int32).to(cuda_device)
                dispatched_input = torch.randn(1, N*len(indices),  self.config.hidden_size,
                                        dtype=torch.bfloat16).to(cuda_device)
                with torch.no_grad():
                    _, _ = model(dispatched_input, tokens_per_expert)
        print("  experts finished warmup")
    
    def update_replicated_experts (self, layer_id,replica_count_per_index ):
        '''    
        change replicated experts for a specific layer
        '''        
        # Expand replica count dict to list of global expert indices
        expert_indices  = []
        for expert_id, cnt in replica_count_per_index.items():
            expert_indices.extend( [expert_id] * cnt )
        print(f"Layer {layer_id} new expert_indices", expert_indices)
        # Skip update if no change
        if expert_indices  == self.expert_indices[layer_id]:
            print(f"Layer {layer_id} skip update replicated experts weight")
            return 
        # NOTE: Default: indices are sorted for consistency
        if len(expert_indices) == len(  self.expert_indices[layer_id] ):
            # Replace expert weights only
            expert_model = self.experts[layer_id]
            for idx in range(len(expert_indices)):
                new_id = expert_indices[idx]
                old_id =  self.expert_indices[layer_id] [idx]
                if new_id != old_id:
                    with torch.no_grad():
                        weight_fc1 =  self.cpu_expert_state[f"L{layer_id}_E{new_id}_W1"]
                        weight_fc2 =  self.cpu_expert_state[f"L{layer_id}_E{new_id}_W2"]
                        # Assign weights to the model
                        getattr(expert_model.linear_fc1, f"weight{idx}").data.copy_(weight_fc1)
                        getattr(expert_model.linear_fc2, f"weight{idx}").data.copy_(weight_fc2)
        else:
            # Rebuild new expert module if expert count changed
            expert_module = TEGroupedMLP
            expert_submodule = MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear)
            submodules_experts = ModuleSpec(module=expert_module, submodules=expert_submodule)
            new_expert_model = build_module(submodules_experts, len(expert_indices), self.config)
            try:
                load_expert_weights(new_expert_model, expert_indices, layer_id, self.cpu_expert_state)
              
            except Exception as e:
                print(f"[Error] Failed to load weights for layer {layer_id}: {e}")
                import traceback
                traceback.print_exc()
            self.experts[layer_id] = new_expert_model
        # Update internal state
        self.expert_indices[layer_id] = expert_indices
# Instantiate the expert manager
expert_manager = ExpertsManager()

@app.route("/forward", methods=["POST"])
def forward():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    print("Content-Type:", request.content_type)
    print("Request headers:", request.headers)
    print("Request content-type:", request.content_type)
    print("Request form:", request.form)
    print("Request files:", request.files)
    print("Raw data:", request.get_data())
    
    
    # dispatched_input = request.json["dispatched_input"]
    # tokens_per_expert = request.json["tokens_per_expert"]
    # target_layer = int(request.json["layer"]) if request.json["layer"]!=None else 0

    # dispatched_input = torch.tensor(
    #     request.json["dispatched_input"], dtype=torch.bfloat16).to(cuda_device)
    # tokens_per_expert = torch.tensor(
    #     request.json["tokens_per_expert"], dtype=torch.int32).to(cuda_device)
    
        
    target_layer = int(request.form["layer"]) if request.form["layer"] else 0   
    print("we have target_layer")
    dispatched_input_meta = json.loads(request.form['dispatched_input_meta']) 
    tokens_per_expert_meta = json.loads(request.form['tokens_per_expert_meta']) 
    print("we have two tensors")

    handler = request.files['handler'].read()
    print("we have the handler")
    
    handle_manager = IPCHandleManager(handler, dispatched_input_meta['device'])

    dispatched_input = tensor_restore_from_handler_pybind(handle_manager, dispatched_input_meta)
    tokens_per_expert = tensor_restore_from_handler_pybind(handle_manager, tokens_per_expert_meta)
    
    

    
    updated = expert_manager.update_events[target_layer].wait(timeout=2.0)
    
    if not updated:
        return jsonify({"error": "Expert update not completed in time"}), 503
    with torch.no_grad():
        start_event.record()
        output, _ = expert_manager.experts[target_layer](dispatched_input, tokens_per_expert)
        end_event.record()
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event)
    print(f"Layer {target_layer} forward time", latency_ms, " ms")
    return jsonify({"hidden_output": output.cpu().tolist(), "latency_ms": latency_ms})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "OK"})

@app.route("/pred_workload", methods=["POST"])
def receive_pred_workload():
    try:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        data = request.json
        target_layer = int(data["layer"])
        replica_count_per_index = {
                int(k): v for k, v in data["replica_count_per_index"].items()
            }
        print( "target_layer:", target_layer," ",replica_count_per_index)
        # 清除事件，防止 forward 提前执行
        expert_manager.update_events[target_layer].clear()
        expert_manager.update_replicated_experts(target_layer, replica_count_per_index)
        expert_manager.update_events[target_layer].set()  # 更新完毕，通知 forward
        
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        print(f"  Layer {target_layer} update replicated experts weight time", latency_ms, " ms")
        return jsonify({"status": "ok", "layer": target_layer})

    except Exception as e:
        print(f"[Container] Error in /pred_workload: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
if __name__ == "__main__":
    # assert False, "we got a big problem, different layer should be separated "
    if WARMUP and not DEBUG:
        expert_manager._warmup()
    app.run(host="0.0.0.0", port=5000,threaded=True)
