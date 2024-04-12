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

