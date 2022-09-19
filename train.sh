export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $ROOT_DIR
BASE_PATH=/apdcephfs/share_916081/xfbai/data

dataset=wiki_0_100

datapath=${BASE_PATH}/GraphPLM/${dataset}
MODEL=${BASE_PATH}/bart-base
cache=${datapath}/.cache/

lr=3e-5

OUTPUT_DIR=${BASE_PATH}/outputs/exp.GraphPLM/GraphPLM-bart-base-lr-${lr}-bsz256

if [ ! -d ${OUTPUT_DIR} ];then
  mkdir -p ${OUTPUT_DIR}
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

export HF_DATASETS_CACHE=$cache

torchrun --nnodes=1 --nproc_per_node=8 --max_restarts=0 --rdzv_id=1 --rdzv_backend=c10d main.py \
    --train_file $datapath/train.jsonl \
    --validation_file $datapath/val.jsonl \
    --test_file $datapath/test.jsonl \
    --output_dir $OUTPUT_DIR \
    --cache_dir $cache \
    --model_name_or_path $MODEL \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --mlm_prob 0.15 \
    --smart_init True \
    --learning_rate $lr \
    --optim "adamw_hf" \
    --lr_scheduler_type "polynomial" \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --early_stopping 10 \
    --max_source_length 512 \
    --max_target_length 512 \
    --label_smoothing_factor 0 \
    --evaluation_strategy "epoch" \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --max_steps -1 \
    --use_fast_tokenizer False \
    --logging_dir $OUTPUT_DIR/logs \
    --logging_first_step True \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed 42 \
    --fp16 \
    --fp16_backend "auto" \
    --dataloader_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --remove_unused_columns False \
    --greater_is_better False \
    --do_train \
    --do_eval \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OUTPUT_DIR/run.log
