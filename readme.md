# Look, Compare, Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning

This repository is the official implementation of "Look, Compare, Decide: Alleviating Hallucination in Large Vision-Language Models via Multi-View Multi-Path Reasoning". 

## Pipeline of MVP

![image](https://img2.imgtp.com/2024/04/12/bGmsicds.png)


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
This will create a file under the `output` folder that stores all the captions. Of course, you need to execute `(bottom-up, top-down, normal)` separately under the `perspective` parameter.

**We have prepared the caption file and can use it directly in the `output` folder.**

### MVP
To employ MVP, you can use the following command in `MVP_llava.sh`:
```bash
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
```
After that, you can obtain the result files in the `output` folder.

### Important arguments
- `--perspective`: the caption perspective.
- `--topk`: employ topk's reasoning paths.
- `--threshold`: If logits is less than this value, filter out current reasoning path in order to speed the inference.


## Evaluation

To evaluate the performance of MVP, you can use the following command in `eval_pope.sh`:
```bash
python eval/eval_pope.py \
    --gt_files MVP/benchmark/POPE/coco/coco_pope_popular.json \
    --gen_files_bottom_up MVP/output/MVP_llava_bottom-up_coco_popular_pope.jsonl \
    --gen_files_top_down MVP/output/MVP_llava_top-down_coco_popular_pope.jsonl \
    --gen_files_normal MVP/output/MVP_llava_normal_coco_popular_pope.jsonl \
    --a 0.4 --b 0.4 --c 0.2
```

### Important arguments
- `--a`: the weight of bottom-up path.
- `--b`: the weight of top-down path.
- `--c`: the weight of normal path.


## Experiment Results

MVP's performance on POPE:

![image](https://img2.imgtp.com/2024/04/12/xuzu2nd1.png)

MVP's performance on MME:

![image](https://img2.imgtp.com/2024/04/12/vfI5Zjvb.png)



## Case Study

![image](https://img2.imgtp.com/2024/04/12/RyxiIyea.png)
![image](https://img2.imgtp.com/2024/04/12/yGWrU9oL.png)



## Contributing

MIT License

Copyright (c) [2024] [anonymous]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.