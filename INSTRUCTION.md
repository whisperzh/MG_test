# Step 1 prepare weight
```python
from safetensors.torch import load_file
import torch

file_path1 = '/home/ubuntu/MG_test/checkpoints/model-00001-of-00019.safetensors' # replace with your actual file path
file_path2 = '/home/ubuntu/MG_test/checkpoints/model-00002-of-00019.safetensors' # replace with your actual file path
loaded = {}
loaded.update(load_file(file_path1))
loaded.update(load_file(file_path2))
loaded.keys()

import os
save_weights_dir="/home/ubuntu/MG_test/weights/"
for layer in range(2):
    for expert in range(8):
        w1_prefix = f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight'
        w2_prefix = f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight'
        w3_prefix = f'model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight'
        target_prefix_fc1 = f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert}"
        target_prefix_fc2 = f"decoder.layers.{layer}.mlp.experts.linear_fc2.weight{expert}"
        target_prefix_fc1=save_weights_dir+target_prefix_fc1.replace(".","_")+".pt"
        target_prefix_fc2=save_weights_dir+target_prefix_fc2.replace(".","_")+".pt"
        fc1= torch.cat([
                loaded[w1_prefix],
                loaded[w3_prefix],
            ], dim=0)
        
        fc2= loaded[w2_prefix]
        
        torch.save(fc1,target_prefix_fc1)
        torch.save(fc2,target_prefix_fc2)

```

# Step 2 Create Container
```shell
cd megatron/expert_replicate &&
sudo docker build -t expert_container .
```
# Step 3 Launch Megatron framework
- Before you run the following code please remember to change 
    1. master_addr
    2. EXPERTS_ADDRESS
    3. EXTERNAL_EXPERTS
    4. EXPERTS_COPY
- You also probably need to change the port in moe_layer.py (line 399)

```shell
source venv/bin/activate &&
cd mixtral/REPLICATE/ &&
sh infer.sh
```
# Step 4 Launch expert container instances
- you have 2 options:
either use shell
```shell
docker run -d --name moe_layer_0_exp_0_3 --gpus all --rm -p 5001:5000 \
  -v /home/ubuntu/MG_test/weights:/app/weights \
  -v /home/ubuntu/MG_test/mixtral/REPLICATE/saved_objects:/app/saved_objects \
  -e RANK=0 \
  -e "EXPERTS=[[0, 1, 2, 3], [4, 5, 6, 7]]" \
  -e GPU_IDX=0 \
  -e WEIGHT_PATH=/app/weights \
  -e "LAYER=[0]" \
  -e PATH_SAVEDOBJ=/app/saved_objects \
  expert_container

docker run -d --name moe_layer_0_exp_4_7 --gpus all --rm -p 5002:5000 \
  -v /home/ubuntu/MG_test/weights:/app/weights \
  -v /home/ubuntu/MG_test/mixtral/REPLICATE/saved_objects:/app/saved_objects \
  -e RANK=1 \
  -e "EXPERTS=[[0, 1, 2, 3], [4, 5, 6, 7]]" \
  -e GPU_IDX=0 \
  -e WEIGHT_PATH=/app/weights \
  -e "LAYER=[0]" \
  -e PATH_SAVEDOBJ=/app/saved_objects \
  expert_container
```
or
use python code
```python
import subprocess


NUM_CONTAINERS = 2
IMAGE_NAME = "expert_container"
BASE_PORT = 5001
layer = [0]


experts=[[0,1,2,3],[4,5,6,7]]
for i in range(NUM_CONTAINERS):
    cmd = [
    "docker", "run", "-d",
    "--name", f"moe_layer_{layer[0]}_exp_{experts[i][0]}_{experts[i][-1]}",
    "--gpus", "all",
    "--rm",
    "-p", f"{BASE_PORT+i}:5000",
    "-v", "/home/ubuntu/MG_test/weights:/app/weights",
    "-v", "/home/ubuntu/MG_test/mixtral/REPLICATE/saved_objects:/app/saved_objects",
    "-e", f"RANK={i}",
    "-e", f"EXPERTS={experts}",
    "-e", f"GPU_IDX={0}",
    "-e", f"WEIGHT_PATH=/app/weights",
    "-e", f"LAYER={layer}",
    "-e", f"PATH_SAVEDOBJ=/app/saved_objects",
    
    IMAGE_NAME
    ]
    try:
        # with open('launch_instances.sh', 'a', encoding='utf-8') as file:
        #     file.write(' '.join(cmd))
        subprocess.run(cmd, check=True)
        print(f"moe_layer_exp_{experts[i][0]}_{experts[i][-1]}\n容器启动成功！")
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
```