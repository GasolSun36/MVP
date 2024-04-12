import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import copy

sys.path.append('MVP/model')

from model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.conversation import conv_templates, SeparatorStyle
from model.llava.model.builder import load_pretrained_model
from model.llava.utils import disable_torch_init
from model.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
from transformers import set_seed

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def check_subset(string):
    word_list = ['yes', 'no', 'Yes', "No"]
    word_string = string.split(' ')
    for word in word_list:
        for token in word_string:
            if word == token:
                return word
    return None

def get_token_id(tokenizer, token, output_ids):
    prefixed_token = '▁' + token
    try:
        return output_ids.index(tokenizer.convert_tokens_to_ids(prefixed_token))
    except ValueError:
        return output_ids.index(tokenizer.convert_tokens_to_ids(token))
    
def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    caption_data=[]
    caption_file = "MVP/output/coco_pope_popular_caption_llava_{}.jsonl".format(args.perspective)

    with open(caption_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            caption_data.append(data)

    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n'

        conv = conv_templates[args.conv_mode].copy()
        image_caption = caption_data[i]['text']

        conv.append_message(conv.roles[0], qs + "Here is the caption of the image:\n{}\nQuestion: ".format(image_caption) + cur_prompt.lower())
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        with torch.inference_mode():
            outputs = model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), return_dict=True)
            predictions = outputs.logits
            probabilities = torch.softmax(predictions[0, -1, :], dim=-1)
            top_k_values, top_k_indices = probabilities.topk(args.topk)

            predicted_tokens = tokenizer.decode(top_k_indices.tolist(), skip_special_tokens=True).strip().split()
            aggregation_dict = {'Yes': 0, 'No': 0}
            
            for token, value in zip(predicted_tokens, top_k_values):
                if token in aggregation_dict:
                    aggregation_dict[token] += value.item()

            
            
            filtered_tokens = [token for token in predicted_tokens if token not in ['Yes', 'No']]
            indices = [i for i, token in enumerate(predicted_tokens) if token not in ['Yes', 'No']]
            filtered_values = top_k_values[indices]
            for token, value in zip(filtered_tokens, filtered_values):
                if value < args.threshold:
                    continue
                new_prompt = prompt + token
                input_ids = tokenizer_image_token(new_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                output = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=args.max_new_tokens,
                    top_p=args.top_p,
                    use_cache=True)
                

                output_ids = output.sequences
                output_scores = list(output.scores)
                outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                answer = check_subset(outputs)

                if answer is None:
                    aggregation_dict['Yes'] += value
                    continue
                
                last_output_ids_list = output_ids[0][input_ids.shape[1]:].tolist()
                answer_index = get_token_id(tokenizer, answer, last_output_ids_list)

                answer_probabilities = torch.softmax(output_scores[answer_index][0], dim=-1)
                top_two_probs = answer_probabilities.topk(2)

                answer_lower = answer.lower()
                if answer_lower in ["yes", "no"]:
                    diff_prob = top_two_probs.values[0] - top_two_probs.values[1]
                    if answer_lower == "yes":
                        yes_aggretion_logits += value * diff_prob
                    elif answer_lower == "no":
                        no_aggretion_logits += value * diff_prob
                else:
                    print(f"Unexpected answer token: {answer}")
                    import ipdb; ipdb.set_trace()

            final_answer = "Yes" if aggregation_dict['Yes'] > aggregation_dict['No'] else "No"
            
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": final_answer,
                                   'aggretion_logits': aggregation_dict,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--perspective", type=str, default="bottom-up")  # bottom-up, top-down, normal
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--caption_file", type=str, default="")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
