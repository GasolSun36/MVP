python vanilla_decoding/llava.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder MVP/data/coco \
    --question-file MVP/benchmark/POPE/coco/coco_pope_popular.json \
    --answers-file MVP/output/llava_vanilla.jsonl \
    --temperature 0.7 \
    --top_p 1.0 \
    --max_new_tokens 5 \
    --num_beams 1 --seed 336

