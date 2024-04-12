python eval/eval_pope.py \
    --gt_files MVP/benchmark/POPE/coco/coco_pope_popular.json \
    --gen_files_bottom_up MVP/output/MVP_llava_bottom-up_coco_popular_pope.jsonl \
    --gen_files_top_down MVP/output/MVP_llava_top-down_coco_popular_pope.jsonl \
    --gen_files_normal MVP/output/MVP_llava_normal_coco_popular_pope.jsonl \
    --a 0.4 --b 0.4 --c 0.2