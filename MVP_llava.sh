#!/bin/bash

declare -a files=("MVP_llava")

declare -a perspectives=("bottom-up" "top-down" "normal")

declare -a question_files=("coco")
declare -a question_types=("popular")

for file in "${files[@]}"; do
  for perspective in "${perspectives[@]}"; do
    for dataset in "${question_files[@]}"; do
      for type in "${question_types[@]}"; do
        question_file="MVP/benchmark/POPE/${dataset}/${dataset}_pope_${type}.json"
        output_file="MVP/output/$(basename "$file" .py)_${perspective}_${dataset}_${type}_pope.jsonl"
        log_file="MVP/logs/$(basename "$file" .py)_${perspective}_${dataset}_${type}_pope.log"

        nohup srun -p -n1 -N1 --gres=gpu:1 --quotatype=reserved python "MVP/$file" \
          --model-path liuhaotian/llava-v1.5-7b \
          --image-folder "MVP/data/${dataset}" \
          --question-file "$question_file" \
          --perspective "$perspective" \
          --answers-file "$output_file" \
          --temperature 0.7 \
          --top_p 1.0 --topk 3 \
          --threshold 0.001 \
          --max_new_tokens 50 \
          --num_beams 1 --seed 336
          1>"$log_file" 2>&1 &

        sleep 3
      done
    done
  done
done
