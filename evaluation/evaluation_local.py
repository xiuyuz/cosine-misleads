import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'train'))

import torch
from datasets import load_dataset
import json
from tqdm import tqdm
from src.model.qwen_lvr_model import QwenWithLVR
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

from qwen_vl_utils import process_vision_info
from PIL import Image
import csv

# ==== Paths ====
# Reads paths from environment variables so you do not need to edit
# the file. Defaults assume a layout under ~/plvr-workspace; override by
# exporting WORKSPACE and DATA_ROOT in your shell, or pass --ckpt to
# run_one_ckpt.py to bypass CHKPT_PATHS entirely.
WORKSPACE = os.environ.get("WORKSPACE", os.path.expanduser("~/plvr-workspace"))
DATA_ROOT = os.environ.get("DATA_ROOT", WORKSPACE)

HF_HOME = os.path.join(DATA_ROOT, "huggingface")
os.environ["HF_HOME"] = HF_HOME

EVAL_DIR = os.path.join(WORKSPACE, "eval_data")
RESULTS_DIR = os.path.join(WORKSPACE, "eval_results")

CHKPT_PATHS = [
    os.path.join(WORKSPACE, "plvr2/checkpoints/stage1/checkpoint-2500"),
    os.path.join(WORKSPACE, "plvr3/checkpoints/stage1/checkpoint-2500"),
]

# For LVR baseline: total token budget per sequence
# For P-LVR: per-stage budget (ctx stage + tgt stage each get this many tokens)
# With expansion_factor=1.5, ctx has ~2.25x more tokens than tgt in training.
# Use PLVR_STEP_LIST (roughly 2x) to match training distribution for ctx stage.
STEP_LIST = [4, 8, 16]           # LVR baseline / P-LVR total-token-matched comparison
PLVR_STEP_LIST = [8, 16, 32]     # P-LVR per-stage budget matched to ctx training distribution
DECODING_STRATEGY = "steps"
PLVR_TARGET_ONLY = False
RESULTS_TAG = DECODING_STRATEGY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


DEFAULT_EVAL_BATCH_SIZE = _get_env_int("EVAL_BATCH_SIZE", 1)
BLINK_EVAL_BATCH_SIZE = _get_env_int("BLINK_EVAL_BATCH_SIZE", DEFAULT_EVAL_BATCH_SIZE)
VSTAR_EVAL_BATCH_SIZE = _get_env_int("VSTAR_EVAL_BATCH_SIZE", DEFAULT_EVAL_BATCH_SIZE)
MMVP_EVAL_BATCH_SIZE = _get_env_int("MMVP_EVAL_BATCH_SIZE", DEFAULT_EVAL_BATCH_SIZE)


def get_results_tag(decoding_strategy, is_plvr):
    if is_plvr and PLVR_TARGET_ONLY and decoding_strategy == "steps":
        return "steps_tgt_only"
    return decoding_strategy

DATASET_CONFIG = {
    'blink': {
        "loader": lambda gen_w_head, run_name, decoding_strategy: load_blink_dataset(gen_w_head, run_name, decoding_strategy),
        "evaluator": lambda model, proc, data, img_dir, out_dir, ds_name, decoding_strategy: evaluate_blink(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
    "vstar": {
        "loader": lambda gen_w_head, run_name, decoding_strategy: load_vstar_dataset(gen_w_head, run_name, decoding_strategy),
        "evaluator": lambda model, proc, data, img_dir, out_dir, ds_name, decoding_strategy: evaluate_vstar(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
    "MMVP": {
        "loader": lambda gen_w_head, run_name, decoding_strategy: load_mmvp_dataset(gen_w_head, run_name, decoding_strategy),
        "evaluator": lambda model, proc, data, img_dir, out_dir, ds_name, decoding_strategy: evaluate_mmvp(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
}

# ==== Core utilities ====

def accuracy_reward(response: str, ground_truth: str) -> float:
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer == ground_truth

def get_task_instruction(bench_name):
    if bench_name in ("vstar", "mmvp", "blink"):
        return "\nAnswer with the option's letter from the given choices directly."
    raise ValueError(f"Unknown benchmark: {bench_name}")

def create_messages(img_path, question):
    if not isinstance(img_path, list):
        img_path = [img_path]
    vision_content = [{"type": "image", "image": ip} for ip in img_path]
    vision_content.append({"type": "text", "text": question})
    return [{"role": "user", "content": vision_content}]


def get_eval_batch_size(ds_name):
    ds_name = ds_name.lower()
    if ds_name == "blink":
        return BLINK_EVAL_BATCH_SIZE
    if ds_name == "vstar":
        return VSTAR_EVAL_BATCH_SIZE
    if ds_name == "mmvp":
        return MMVP_EVAL_BATCH_SIZE
    return DEFAULT_EVAL_BATCH_SIZE


def get_batch_items(dataset, start, end):
    end = min(end, len(dataset))
    return [dataset[idx] for idx in range(start, end)]

def load_model_and_processor(chkpt_pth):
    run_name = '_'.join(chkpt_pth.split('/'))
    config = AutoConfig.from_pretrained(chkpt_pth)
    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)
    model = QwenWithLVR.from_pretrained(
        chkpt_pth,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(chkpt_pth)
    return model, processor, run_name

def run_inference(model, processor, img_path, text, steps, decoding_strategy):
    return run_inference_batch(model, processor, [img_path], [text], steps, decoding_strategy)


def run_inference_batch(model, processor, img_paths, texts, steps, decoding_strategy):
    messages = [create_messages(img_path, text) for img_path, text in zip(img_paths, texts)]
    text_formatted = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text_formatted,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to("cuda")
    batch_size = len(texts)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            lvr_steps=[steps] * batch_size,
            plvr_target_only=PLVR_TARGET_ONLY and getattr(model.config, 'plvr_mode', False),
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    return output_text

# ==== Evaluation functions (unchanged logic) ====

def evaluate_vstar(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps"):
    print(f"Evaluating VSTAR with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("vstar")
    batch_size = get_eval_batch_size(ds_name)
    print(f"Using eval batch size: {batch_size}")
    step2results_category = {}
    step2results_overall = {}
    for steps in STEP_LIST:
        step2results_category[steps] = {}
        out_file = os.path.join(out_dir, f"{decoding_strategy}{steps:03d}.json")
        total, correct = 0, 0
        res_by_category = {}
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                if "category" not in res:
                    res["category"] = "direct_attributes" if int(res["id"]) <= 114 else "relative_position"
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total": 0, "correct": 0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            result = []
            for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating Vstar, steps={steps}"):
                batch = get_batch_items(dataset, start, start + batch_size)
                img_paths = [os.path.join(image_dir, dat['image']) for dat in batch]
                texts = [dat['text'] + task_instruction for dat in batch]
                outputs = run_inference_batch(model, processor, img_paths, texts, steps, decoding_strategy)
                for dat, output in zip(batch, outputs):
                    res = {'id': dat['question_id'], 'prediction': [output], 'label': dat['label'], 'category': dat['category']}
                    result.append(res)
                    if accuracy_reward(output, dat['label']):
                        correct += 1
                    total += 1
            json.dump(result, open(out_file, 'w+'), indent=2)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    for category in ["direct_attributes", "relative_position"]:
        res = []
        for steps in step2results_category:
            rbc = step2results_category[steps]
            if category in rbc:
                res.append(rbc[category]["correct"] / rbc[category]["total"])
        print(f"{category}: " + ",".join([f"{x*100:.2f}" for x in res]))

def evaluate_mmvp(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps"):
    print(f"Evaluating MMVP with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("mmvp")
    batch_size = get_eval_batch_size(ds_name)
    print(f"Using eval batch size: {batch_size}")
    for steps in STEP_LIST:
        out_file = os.path.join(out_dir, f"{decoding_strategy}{steps:03d}.json")
        total, correct = 0, 0
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                total += 1
        else:
            result = []
            for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating MMVP, steps={steps}"):
                batch = get_batch_items(dataset, start, start + batch_size)
                img_paths = []
                texts = []
                labels = []
                for dat in batch:
                    img = dat['image']
                    if isinstance(img, Image.Image):
                        img_path = img
                    else:
                        img_path = os.path.join(image_dir, img)
                    label = dat['label']
                    if label in ['(a)', '(b)']:
                        label = label.strip().upper()[1]
                    img_paths.append(img_path)
                    texts.append(dat['query'].replace('(a)', 'A.').replace('(b)', 'B.') + task_instruction)
                    labels.append(label)
                outputs = run_inference_batch(model, processor, img_paths, texts, steps, decoding_strategy)
                for dat, output, label in zip(batch, outputs, labels):
                    res = {'id': dat['question_id'], 'prediction': [output], 'label': label}
                    result.append(res)
                    if accuracy_reward(output, label):
                        correct += 1
                    total += 1
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
        json.dump(result, open(out_file, 'w+'), indent=2)

def evaluate_blink(model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy="steps"):
    print(f"Evaluating BLINK with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir, exist_ok=True)
    task_instruction = get_task_instruction("blink")
    batch_size = get_eval_batch_size(ds_name)
    print(f"Using eval batch size: {batch_size}")
    step2results_category = {}
    step2results_overall = {}
    for steps in STEP_LIST:
        step2results_category[steps] = {}
        out_file = os.path.join(out_dir, f"{decoding_strategy}{steps:03d}.json")
        total, correct = 0, 0
        res_by_category = {}
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                result = json.load(f)
            for res in result:
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total": 0, "correct": 0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            result = []
            for start in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating BLINK, steps={steps}"):
                batch = get_batch_items(dataset, start, start + batch_size)
                img_paths = [dat['image'] for dat in batch]
                texts = [dat['query'] + task_instruction for dat in batch]
                outputs = run_inference_batch(model, processor, img_paths, texts, steps, decoding_strategy)
                for dat, output in zip(batch, outputs):
                    res = {'id': dat['question_id'], 'prediction': [output], 'label': dat['label'], 'category': dat['category']}
                    result.append(res)
                    if accuracy_reward(output, dat['label']):
                        correct += 1
                    total += 1
            json.dump(result, open(out_file, 'w+'), indent=2)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    for category in ['Counting', 'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
                     'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
                     'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']:
        res = []
        for steps in step2results_category:
            rbc = step2results_category[steps]
            if category in rbc:
                res.append(rbc[category]["correct"] / rbc[category]["total"])
        print(category + ',' + ",".join([f"{x*100:.2f}" for x in res]))

# ==== Data Loaders ====

import string

def load_vstar_dataset(gen_w_head, run_name, decoding_strategy):
    from huggingface_hub import snapshot_download
    image_dir = os.path.join(EVAL_DIR, "vstar_bench")
    if not os.path.exists(image_dir):
        print("V*Star images not found locally, downloading from HuggingFace...")
        snapshot_download(
            repo_id="craigwu/vstar_bench",
            repo_type="dataset",
            local_dir=image_dir,
            ignore_patterns=["*.parquet", "*.arrow", "*.json"],
        )
        print(f"V*Star images downloaded to {image_dir}")
    ds = load_dataset("craigwu/vstar_bench")
    out_dir = os.path.join(RESULTS_DIR, "vstar", f"decoding_by_{RESULTS_TAG}", run_name)
    return ds['test'], image_dir, out_dir, "vstar"

def load_mmvp_dataset(gen_w_head, run_name, decoding_strategy):
    from huggingface_hub import hf_hub_download, snapshot_download
    mmvp_dir = os.path.join(EVAL_DIR, "MMVP")
    csv_file = os.path.join(mmvp_dir, "Questions.csv")
    image_dir = os.path.join(mmvp_dir, "MMVP_Images")

    if not os.path.exists(csv_file):
        print("MMVP not found locally, downloading from HuggingFace...")
        os.makedirs(mmvp_dir, exist_ok=True)
        # Download Questions.csv directly from the HF repo
        hf_hub_download(
            repo_id="MMVP/MMVP",
            repo_type="dataset",
            filename="Questions.csv",
            local_dir=mmvp_dir,
        )
        print(f"MMVP CSV downloaded to {mmvp_dir}")

    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print("MMVP images not found locally, downloading from HuggingFace...")
        os.makedirs(image_dir, exist_ok=True)
        ds = load_dataset("MMVP/MMVP")
        for split in ds:
            for i, item in enumerate(ds[split]):
                img_path = os.path.join(image_dir, f"{i+1}.jpg")  # 1-indexed to match CSV
                if not os.path.exists(img_path):
                    item['image'].save(img_path)
        print(f"MMVP images downloaded to {image_dir}")

    data = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["Index"])
            data.append({
                "question_id": idx,
                'image': f"{idx}.jpg",
                "query": row["Question"] + '\nOptions:\n' + row["Options"],
                "label": row["Correct Answer"]
            })
    out_dir = os.path.join(RESULTS_DIR, "MMVP", f"decoding_by_{RESULTS_TAG}", run_name)
    return data, image_dir, out_dir, "mmvp"

def load_blink_dataset(gen_w_head, run_name, decoding_strategy):
    configs = ['Counting', 'IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
    processed_data = []
    for config in configs:
        ds = load_dataset("BLINK-Benchmark/BLINK", config)['val']
        for dat in ds:
            choices = dat["choices"]
            letters = string.ascii_uppercase
            option_string = "".join(f"{l}. {c}\n" for l, c in zip(letters, choices))
            ans = dat['answer'][1].upper() if len(dat['answer']) > 1 else dat['answer'][0].upper()
            images = [dat[k] for k in ['image_1', 'image_2', 'image_3', 'image_4'] if k in dat and dat[k] is not None]
            processed_data.append({
                "question_id": dat["idx"],
                "image": images,
                "query": dat['question'] + "\nOptions:\n" + option_string,
                "label": ans,
                "category": config
            })
    out_dir = os.path.join(RESULTS_DIR, "blink", f"decoding_by_{RESULTS_TAG}", run_name)
    return processed_data, None, out_dir, "blink"

# ==== Main ====

def main():
    global STEP_LIST, RESULTS_TAG
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for checkpoint_dir in CHKPT_PATHS:
        model, processor, run_name = load_model_and_processor(checkpoint_dir)
        gen_w_head = model.config.lvr_head
        is_plvr = getattr(model.config, 'plvr_mode', False)
        STEP_LIST = PLVR_STEP_LIST if is_plvr else [4, 8, 16]
        RESULTS_TAG = get_results_tag(DECODING_STRATEGY, is_plvr)
        print(
            "\n" + "="*80
            + f"\nEvaluating: {checkpoint_dir} | plvr={is_plvr} | target_only={PLVR_TARGET_ONLY and is_plvr} | steps={STEP_LIST} | results_tag={RESULTS_TAG}\n"
            + "="*80
        )
        for bench_name, cfg in DATASET_CONFIG.items():
            print(f"\n{'<'*32} {bench_name} {'>'*32}")
            dataset, image_dir, out_dir, ds_name = cfg["loader"](gen_w_head, run_name, DECODING_STRATEGY)
            cfg["evaluator"](model, processor, dataset, image_dir, out_dir, ds_name, DECODING_STRATEGY)

if __name__ == "__main__":
    main()
