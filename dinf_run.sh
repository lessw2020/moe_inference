# export CUDA_VISIBLE_DEVICES=4,5,6,7
PORT=${1:-29501}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0,6,7}
torchrun --nproc-per-node=$NGPU --master_port=$PORT \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
dist_run.py llama3 --pp 1
