set -ex
# models
RESULTS_DIR='./results/sandstone'

# dataset
CLASS='sandstone'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=128 # scale images to this size
FINE_SIZE=128 # then crop to this size
INPUT_NC=1  # number of channels in the input image

# misc
GPU_ID=1   # gpu id
NUM_TEST=10 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
