export CUDA_VISIBLE_DEVICES=$1
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

Dataset=LDC2020

BasePath=/apdcephfs/share_916081/xfbai/data
DataPath=$BasePath/AMR-Processed/$Dataset

ModelCate=bart-base

MODEL=$BasePath/$ModelCate
MODEL=/apdcephfs/share_916081/xfbai/data/output/exp.GraphPLM/wiki_0_100_new-GraphPLM-bart-base-lr-3e-5-bsz256/checkpoint-178990
MODEL=/apdcephfs/share_916081/xfbai/data/output/exp.GraphPLM/wiki_amr_full-GraphPLM-bart-base-lr-3e-5-bsz256-new/checkpoint-582910

ModelCache=$BasePath/.cache
DataCache=$DataPath/.cache/dump-amr2text

lr=$2

OutputDir=${BasePath}/output/exp.PLMGen/$Dataset-$ModelCate-WikiAMRFull-AMR2Text-bsz8-lr-${lr}-1028-AMRToken-nosmart

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
    --num_train_epochs 20 \
    --early_stopping 10 \
    --max_source_length 1024 \
    --max_target_length 384 \
    --val_max_target_length 384 \
    --generation_max_length 380 \
    --generation_num_beams 5 \
    --label_smoothing_factor 0.1 \
    --evaluation_strategy "epoch" \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --max_steps -1 \
    --predict_with_generate \
    --smart_init False \
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
    --eval_dataloader_num_workers 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_bleu" \
    --greater_is_better True \
    --do_train \
    --do_eval \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run.log
