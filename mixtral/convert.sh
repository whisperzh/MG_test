TOKENIZER_MODEL=/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1/tokenizer.model
MEGATRON_PATH=/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE="1"
TARGET_EP_SIZE="1"
TARGET_PP_SIZE="4"
 
HF_FORMAT_DIR=/home/ec2-user/CodeSpace/download/models/Mixtral-8x7B-v0.1
MEGATRON_FORMAT_DIR=/home/ec2-user/CodeSpace/NEW_Megatron/Megatron-LM-core_v0.12.0/mixtral/mixtral-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}Layer1

python ../tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf  \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL} \
--saver-transformer-impl transformer_engine \
 
 
 