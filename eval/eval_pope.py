import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_files", type=str, default="MVP/benchmark/POPE/coco/coco_pope_popular.json")
    parser.add_argument("--gen_files_bottom_up", type=str, default="MVP/output/MVP_llava_bottom-up_coco_popular_pope.jsonl")
    parser.add_argument("--gen_files_top_down", type=str, default="MVP/output/MVP_llava_top-down_coco_popular_pope.jsonl")
    parser.add_argument("--gen_files_normal", type=str, default="MVP/output/MVP_llava_normal_coco_popular_pope.jsonl")
    parser.add_argument("--a", type=float, default=0.4)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--c", type=float, default=0.2)
    return parser.parse_args()

def load_json_lines(file_path):
    try:
        with open(os.path.expanduser(file_path), 'r') as file:
            return [json.loads(line) for line in file]
    except Exception as e:
        print(f"Failed to load or parse {file_path}: {e}")
        return []

def compute_probabilities(weights, *args):
    return sum(weight * data['aggretion_logits'][key] for weight, data, key in zip(weights, args, ['Yes', 'No']))

def evaluate_predictions(args, gt_data, predictions):
    true_pos, true_neg, false_pos, false_neg, unknown, yes_answers = 0, 0, 0, 0, 0, 0
    weights = (args.a, args.b, args.c)
    
    for gt, bottom_up, top_down, normal in zip(gt_data, *predictions):
        gt_answer = gt['label'].lower().strip()
        yes_prob, no_prob = compute_probabilities(weights, bottom_up, top_down, normal)
        gen_answer = 'yes' if yes_prob > no_prob else 'no'
        
        if gt_answer in {'yes', 'no'}:
            if gen_answer == gt_answer:
                true_pos += 1 if gen_answer == 'yes' else true_neg
            else:
                false_neg += 1 if gt_answer == 'yes' else false_pos
            if gen_answer == 'yes':
                yes_answers += 1
        else:
            unknown += 1
    
    total_questions = len(gt_data)
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = (true_pos + true_neg) / total_questions if total_questions > 0 else 0
    yes_proportion = yes_answers / total_questions if total_questions > 0 else 0
    unknown_prop = unknown / total_questions if total_questions > 0 else 0
    
    return accuracy, precision, recall, f1, yes_proportion, unknown_prop

def main():
    args = parse_args()
    gt_files = load_json_lines(args.gt_files)
    gen_files_bottom_up = load_json_lines(args.gen_files_bottom_up)
    gen_files_top_down = load_json_lines(args.gen_files_top_down)
    gen_files_normal = load_json_lines(args.gen_files_normal)
    
    results = evaluate_predictions(args, gt_files, (gen_files_bottom_up, gen_files_top_down, gen_files_normal))
    
    print(f"Accuracy: {results[0]*100:.2f}%")
    print(f"Precision: {results[1]*100:.2f}%")
    print(f"Recall: {results[2]*100:.2f}%")
    print(f"F1 Score: {results[3]*100:.2f}%")
    print(f"Yes Proportion: {results[4]*100:.2f}%")
    print(f"Unknown Proportion: {results[5]*100:.2f}%")

if __name__ == "__main__":
    main()
