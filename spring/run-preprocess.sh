# !/bin/bash

cate=silver
echo "Preprocessing $cate ..."
mkdir -p ../data/$cate					# Output Linearized AMR Path
CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 
