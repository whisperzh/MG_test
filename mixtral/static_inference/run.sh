export MOE_TIME=0
export IDEAL=0
DISTRIBUTED_ARGS="--nproc_per_node 4"
                #   --nnodes 1 \
                #   --node_rank 0 \
                #   --master_addr localhost \
                #   --master_port 6000"
CHECKPOINT="/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/mixtral-mcore-TP1PP1EP4Layer1"
TOKENIZER_MODEL=/home/ec2-user/CodeSpace/Megatron-LM/ckp/tokenizer.model

export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun $DISTRIBUTED_ARGS ./gpt_static_inference.py    \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --expert-model-parallel-size 4 \
       --load ${CHECKPOINT}  \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model $TOKENIZER_MODEL \
       --use-mcore-models \
       --max-position-embeddings 32768 \
       --num-layers 1 \
       --hidden-size 4096 \
       --ffn-hidden-size 14336 \
       --num-attention-heads 32 \
       --normalization RMSNorm \
       --disable-bias-linear \
       --position-embedding-type rope \
       --no-position-embedding \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --group-query-attention \
       --num-query-groups 8 \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --seed 42 \
       --num-experts 8 \
       --moe-router-topk 2 \
       --moe-token-dispatcher-type alltoall \
       --moe-grouped-gemm \
       --rotary-base 1000000 \
       --no-rope-fusion \
       --no-gradient-accumulation-fusion \
       --max-batch-size 8 \
       --inference-max-seq-length 32768 \
       --prompts "prompt one " "sample prompt two" "sample prompt 3 hi hi hi hi" "this is me" \
       --num-tokens-to-generate 1 \
       
 

