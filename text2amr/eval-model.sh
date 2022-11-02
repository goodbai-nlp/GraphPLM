export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

Dataset=LDC2020

BasePath=/apdcephfs/share_916081/xfbai/data
DataPath=$BasePath/AMR-Processed/$Dataset

ModelCate=GraphPLM-wikismall-base
ModelCate=GraphPLM-wikifull-base

MODEL=$1

ModelCache=$BasePath/.cache
DataCache=$DataPath/.cache/dump-amrparsing

lr=$2

OutputDir=${BasePath}/output/exp.PLMGen/Eval-$Dataset-$ModelCate-AMRParsing-bsz8-lr-${lr}-AMRToken

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
else
  read -p "${OutputDir} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OutputDir}; mkdir -p ${OutputDir};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

# torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=1 --rdzv_backend=c10d main.py \
python -u main.py \
    --data_dir $DataPath \
    --train_file $DataPath/train.jsonl \
    --validation_file $DataPath/val.jsonl \
    --test_file $DataPath/test.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --model_name_or_path $MODEL \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate $lr \
    --optim "adamw_hf" \
    --lr_scheduler_type "polynomial" \
    --warmup_steps 200 \
    --num_train_epochs 80 \
    --early_stopping 10 \
    --max_source_length 400 \
    --max_target_length 512 \
    --val_max_target_length 512 \
    --generation_max_length 512 \
    --generation_num_beams 5 \
    --label_smoothing_factor 0.1 \
    --evaluation_strategy "no" \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --max_steps -1 \
    --smart_init \
    --predict_with_generate \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --logging_first_step True \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed 42 \
    --fp16 \
    --fp16_backend "auto" \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics \
    --metric_for_best_model "eval_smatch" \
    --greater_is_better True \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/eval.log
