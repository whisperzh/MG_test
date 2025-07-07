source venv/bin/activate &&
cd mixtral/REPLICATE/ &&
sh infer.sh


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