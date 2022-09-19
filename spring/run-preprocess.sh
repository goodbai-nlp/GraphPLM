# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python -u bin/preprocess.py \
    --start 2100 \
    --end 2124 \
    --config configs/config-silver.yaml \
    --direction amr 2>&1 | tee preprocess-silver.log 
