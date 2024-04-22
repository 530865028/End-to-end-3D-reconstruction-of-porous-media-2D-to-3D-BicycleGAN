set -ex
MODEL='bicycle_gan' 
# dataset details
CLASS='sandstone_20'  # sandstone, facades, day2night, edges2shoes, edges2handbags, maps
NZ=8 
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=128
FINE_SIZE=128
INPUT_NC=1
OUTPUT_NC=1

NITER=200
NITER_DECAY=200
SAVE_EPOCH=25

# training
GPU_ID=1
DISPLAY_ID=1
CHECKPOINTS_DIR=./checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}

# command
python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --use_dropout
