#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Train (multi-modal)
CUDA_VISIBLE_DEVICES=0 python3 -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}" \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/mm_dst|mm/gpt2_dst/data/simmc2_special_tokens.json" \
    --train_data_file="/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/mm_dst|mm/gpt2_dst/data/simmc2.1_dials_dstc11_train_target.txt" \
    --do_train \
    --do_eval --eval_all_checkpoints \
    --eval_data_file="/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/mm_dst|mm/gpt2_dst/data/simmc2.1_dials_dstc11_dev_target.txt" \
    --num_train_epochs=10 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --save_total_limit=1
