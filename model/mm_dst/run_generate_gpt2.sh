#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
CUDA_VISIBLE_DEVICES=0 python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}/checkpoint-95000" \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file=/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/mm_coref/gpt2_dst/data/simmc2.1_dials_dstc11_devtest_predict.txt \
    --path_output="${PATH_DIR}"/results/simmc2.1_dials_dstc11_devtest_predicted.txt

