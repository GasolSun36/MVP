# Look, Compare, Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning

This repository is the official implementation of "Look, Compare, Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning". 

## Pipeline of MVP

![image](https://github.com/GasolSun36/MVP/blob/main/assets/pipeline.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



### Datasets
Datasets are in `MVP/benchmark`. Before inference, you need to download the images into the `MVP/data` folder.

## Image Caption
In MVP framework, we need to caption the image first, and you can use the following command in `caption.sh`:
```bash
python caption/llava_caption.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder MVP/data/coco \
    --question-file MVP/benchmark/POPE/coco/coco_pope_popular.json \
    --answers-file MVP/output/coco_pope_popular_caption_llava_bottom-up.jsonl \
    --perspective bottom-up \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 512 \
    --num_beams 1 --seed 336
```
This will create a file under the `output` folder that stores all the captions. Of course, you need to execute `(bottom-up, top-down, regular)` separately under the `perspective` parameter.

**We have prepared the caption file and can use it directly in the `output` folder.**

### MVP
To employ MVP, you can use the following command in `MVP_llava.sh`:
```bash
#!/bin/bash

declare -a files=("MVP_llava")

declare -a perspectives=("bottom-up" "top-down" "regular")

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
          --max_new_tokens 50 \
          --num_beams 1 --seed 336
          1>"$log_file" 2>&1 &

        sleep 3
      done
    done
  done
done
```
After that, you can obtain the result files in the `output` folder.

### Important arguments
- `--perspective`: the caption perspective.
- `--topk`: employ topk's reasoning paths.


## Evaluation

To evaluate the performance of MVP, you can use the following command in `eval_pope.sh`:
```bash
python eval/eval_pope.py \
    --gt_files MVP/benchmark/POPE/coco/coco_pope_popular.json \
    --gen_files_bottom_up MVP/output/MVP_llava_bottom-up_coco_popular_pope.jsonl \
    --gen_files_top_down MVP/output/MVP_llava_top-down_coco_popular_pope.jsonl \
    --gen_files_regular MVP/output/MVP_llava_regular_coco_popular_pope.jsonl \
    --a 0.4 --b 0.4 --c 0.2
```

### Important arguments
- `--a`: the weight of bottom-up path.
- `--b`: the weight of top-down path.
- `--c`: the weight of regular path.


## Experiment Results

MVP's performance on POPE:

![image](https://github.com/GasolSun36/MVP/blob/main/assets/experiment_pope.png)


MVP's performance on MME:

![image](https://github.com/GasolSun36/MVP/blob/main/assets/experiment_mme.png)




## Case Study

![image](https://github.com/GasolSun36/MVP/blob/main/assets/case_study.png)


