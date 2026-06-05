"""
Stage-wise hidden-state alignment analysis for LVR and P-LVR checkpoints.

This script mirrors the stage-1 training data path closely enough to compare
baseline LVR, 2-stage P-LVR, and 3-stage P-LVR checkpoints on the same sample
set. It supports:

- meta-index data paths used by stage-1 training
- reusable sampled index files
- stage-wise cosine similarity summaries
- strict failure handling when too many samples are invalid
"""

import argparse
import bisect
import copy
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    LVR_PLACEHOLDER,
    LVR_TOKEN,
    PLVR_CTX_END_TOKEN,
    PLVR_CTX_START_TOKEN,
    PLVR_FREE_END_TOKEN,
    PLVR_FREE_START_TOKEN,
    PLVR_FREE_TOKEN,
    PLVR_TGT_END_TOKEN,
    PLVR_TGT_START_TOKEN,
    SYSTEM_MESSAGE,
)
from src.dataset.data_utils import get_image_info, llava_to_openai_lvr  # noqa: E402
from src.dataset.lvr_sft_dataset_packed import IterableSupervisedDatasetLVR  # noqa: E402
from src.lvr_utils import expand_bbox  # noqa: E402
from src.model.qwen_lvr_model import QwenWithLVR  # noqa: E402
from src.params import DataArguments  # noqa: E402
from src.train.monkey_patch_forward_lvr import (  # noqa: E402
    replace_qwen2_5_with_mixed_modality_forward_lvr,
)
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb  # noqa: E402


def _load_json(path: str) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _dump_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _normalize_output_name(output_name: Optional[str], config_name: str) -> str:
    if output_name:
        return output_name if output_name.endswith(".json") else f"{output_name}.json"
    return f"alignment_{config_name}.json"


def _normalize_loaded_sample_ids(payload: Any) -> List[int]:
    if isinstance(payload, dict):
        payload = payload.get("sample_ids")
    if not isinstance(payload, list):
        raise ValueError("Sample-id file must contain a list or {'sample_ids': [...]} payload.")
    sample_ids = [int(idx) for idx in payload]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample-id file contains duplicate indices.")
    return sample_ids


def _load_data_sources(data_path: str, image_folder: Optional[str]) -> List[Dict[str, str]]:
    payload = _load_json(data_path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unsupported data payload in {data_path}")

    first_item = payload[0]
    if isinstance(first_item, dict) and "data_path" in first_item and "image_folder" in first_item:
        sources: List[Dict[str, str]] = []
        for item in payload:
            sources.append(
                {
                    "ds_name": item.get("ds_name", os.path.splitext(os.path.basename(item["data_path"]))[0]),
                    "data_path": item["data_path"],
                    "image_folder": item["image_folder"],
                }
            )
        return sources

    return [
        {
            "ds_name": os.path.splitext(os.path.basename(data_path))[0],
            "data_path": data_path,
            "image_folder": image_folder or "",
        }
    ]


def _build_data_args(args: argparse.Namespace) -> DataArguments:
    plvr_mode = args.config in {"2stage", "3stage"}
    include_free_stage = args.config == "3stage"
    return DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        image_min_pixels=128 * 28 * 28,
        image_max_pixels=5120 * 28 * 28,
        random_seed=None,
        plvr_mode=plvr_mode,
        include_free_stage=include_free_stage,
        expansion_factor=1.5,
        num_free_tokens=6,
    )


def _normalize_stage_sizes(stage_sizes: Optional[Sequence[Any]]) -> Optional[List[List[int]]]:
    if stage_sizes is None:
        return None
    if len(stage_sizes) == 1 and isinstance(stage_sizes[0], list) and stage_sizes[0] and isinstance(stage_sizes[0][0], list):
        stage_sizes = stage_sizes[0]
    normalized: List[List[int]] = []
    for item in stage_sizes:
        normalized.append([int(value) for value in item])
    return normalized


def _prepare_model(args: argparse.Namespace, device: str) -> Tuple[QwenWithLVR, AutoProcessor]:
    print(f"Loading model from {args.ckpt}...")
    config = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
    config.latent_end_token = False
    config.lvr_head = False

    replace_qwen2_5_with_mixed_modality_forward_lvr(coconut=True, lvr_head=False)
    replace_qwen_2_5_vl_patch_emb()

    model = QwenWithLVR.from_pretrained(
        args.ckpt,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
    )
    model.eval().to(device)

    processor = AutoProcessor.from_pretrained(args.ckpt)
    tokenizer_vocab = processor.tokenizer.get_vocab()

    model.config.lvr_id = processor.tokenizer.convert_tokens_to_ids(LVR_TOKEN)
    model.config.loss_lvr_fct = getattr(model.config, "loss_lvr_fct", "mse")
    model.config.plvr_mode = args.config in {"2stage", "3stage"}
    model.config.include_free_stage = args.config == "3stage"
    model.config.num_free_tokens = getattr(model.config, "num_free_tokens", 6)

    special_token_map = {
        "ctx_start_id": PLVR_CTX_START_TOKEN,
        "ctx_end_id": PLVR_CTX_END_TOKEN,
        "free_start_id": PLVR_FREE_START_TOKEN,
        "free_end_id": PLVR_FREE_END_TOKEN,
        "free_id": PLVR_FREE_TOKEN,
        "tgt_start_id": PLVR_TGT_START_TOKEN,
        "tgt_end_id": PLVR_TGT_END_TOKEN,
    }
    for attr_name, token in special_token_map.items():
        if token in tokenizer_vocab:
            setattr(model.config, attr_name, processor.tokenizer.convert_tokens_to_ids(token))

    return model, processor


def _resolve_image_paths(raw_source: Dict[str, Any], image_folder: str) -> List[Any]:
    image_files = raw_source["image"]
    if isinstance(image_files, str):
        image_files = [image_files]

    images: List[Any] = []
    for image_file in image_files:
        resolved_path = image_file
        if not os.path.exists(resolved_path) and not str(image_file).startswith("http") and image_folder:
            resolved_path = os.path.join(image_folder, image_file)
        images.append(
            get_image_info(
                resolved_path,
                raw_source["_dataset"].image_min_pixel,
                raw_source["_dataset"].image_max_pixel,
                raw_source["_dataset"].image_resized_w,
                raw_source["_dataset"].image_resized_h,
            )
        )
    return images


def _materialize_sample(dataset: IterableSupervisedDatasetLVR, sample_idx: int) -> Dict[str, Any]:
    raw_source = copy.deepcopy(dataset.raw_data[sample_idx])
    raw_source["_dataset"] = dataset
    processor = dataset.processor
    videos = None
    grid_key = "image_grid_thw"
    pixel_key = "pixel_values"

    images = _resolve_image_paths(raw_source, dataset.image_folder)

    image_grid_thw = processor(
        text=[""],
        images=images,
        videos=videos,
        padding=False,
        do_resize=False,
        return_tensors="pt",
    )["image_grid_thw"]
    lvr_token_idxs_list = dataset.bbox_to_token_idxs(raw_source["bboxes"], image_grid_thw)

    stage_sizes = None
    if dataset.data_args.plvr_mode:
        expanded_bboxes = [expand_bbox(bbox, dataset.data_args.expansion_factor) for bbox in raw_source["bboxes"]]
        lvr_token_idxs_context = dataset.bbox_to_token_idxs(expanded_bboxes, image_grid_thw)

        visual_lvr_tokens: List[List[int]] = []
        all_lvr_tokens: List[List[int]] = []
        stage_sizes = []
        if dataset.data_args.include_free_stage:
            n_free = dataset.data_args.num_free_tokens
            for ctx, tgt in zip(lvr_token_idxs_context, lvr_token_idxs_list):
                visual_lvr_tokens.extend([ctx, tgt])
                all_lvr_tokens.extend([ctx, [0] * n_free, tgt])
                stage_sizes.append([len(ctx), n_free, len(tgt)])
            per_bbox_stage_tokens = [
                (PLVR_CTX_START_TOKEN, LVR_TOKEN, PLVR_CTX_END_TOKEN),
                (PLVR_FREE_START_TOKEN, PLVR_FREE_TOKEN, PLVR_FREE_END_TOKEN),
                (PLVR_TGT_START_TOKEN, LVR_TOKEN, PLVR_TGT_END_TOKEN),
            ]
        else:
            for ctx, tgt in zip(lvr_token_idxs_context, lvr_token_idxs_list):
                visual_lvr_tokens.extend([ctx, tgt])
                all_lvr_tokens.extend([ctx, tgt])
                stage_sizes.append([len(ctx), len(tgt)])
            per_bbox_stage_tokens = [
                (PLVR_CTX_START_TOKEN, LVR_TOKEN, PLVR_CTX_END_TOKEN),
                (PLVR_TGT_START_TOKEN, LVR_TOKEN, PLVR_TGT_END_TOKEN),
            ]

        conversations = copy.deepcopy(raw_source["conversations"])
        stage_tokens = per_bbox_stage_tokens * len(raw_source["bboxes"])
        for conversation in conversations:
            if conversation["from"] == "gpt" and LVR_PLACEHOLDER in conversation["value"]:
                conversation["value"] = conversation["value"].replace(
                    LVR_PLACEHOLDER,
                    LVR_PLACEHOLDER * len(per_bbox_stage_tokens),
                )
        lvr_token_idxs_list = visual_lvr_tokens
        transformed = llava_to_openai_lvr(
            conversations,
            is_video=False,
            lvr_token_idxs_list=all_lvr_tokens,
            latent_end_token=dataset.latent_end_token,
            stage_tokens=stage_tokens,
        )
    else:
        transformed = llava_to_openai_lvr(
            raw_source["conversations"],
            is_video=False,
            lvr_token_idxs_list=lvr_token_idxs_list,
            latent_end_token=dataset.latent_end_token,
        )

    all_input_ids: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_pixel_values: List[torch.Tensor] = []
    all_image_grid_thw: List[torch.Tensor] = []

    if SYSTEM_MESSAGE:
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors="pt")["input_ids"]
        system_labels = torch.full_like(system_ids, IGNORE_INDEX)
        all_input_ids.append(system_ids.squeeze(0))
        all_labels.append(system_labels.squeeze(0))

    for turn_idx in range(0, len(transformed), 2):
        user_input = transformed[turn_idx]
        gpt_response = transformed[turn_idx + 1]

        prompt_text = (
            f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}"
            f"{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
        )
        response_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

        if DEFAULT_IMAGE_TOKEN in prompt_text:
            inputs = processor(
                text=[prompt_text],
                images=images,
                videos=videos,
                padding=False,
                do_resize=False,
                return_tensors="pt",
            )
            prompt_input_ids = inputs["input_ids"]
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])
        else:
            prompt_input_ids = processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"]

        response_input_ids = processor.tokenizer(
            response_text,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )["input_ids"]

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.full((prompt_input_ids.shape[1],), IGNORE_INDEX, dtype=torch.long),
                response_input_ids.squeeze(0),
            ],
            dim=0,
        )

        all_input_ids.append(input_ids.to(torch.long))
        all_labels.append(labels.to(torch.long))

    merged_input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    merged_labels = torch.cat(all_labels, dim=0).to(torch.long)
    attention_mask = (merged_input_ids > -1000000).to(torch.long)

    sample: Dict[str, Any] = {
        "input_ids": merged_input_ids,
        "attention_mask": attention_mask,
        "labels": merged_labels,
        "lvr_tokens": [torch.tensor(group, dtype=torch.long) for group in lvr_token_idxs_list],
        "input_lengths": torch.tensor([merged_input_ids.size(0)], dtype=torch.long),
    }

    if stage_sizes is not None:
        sample["lvr_stage_sizes"] = [stage_sizes]
    if all_pixel_values:
        sample[pixel_key] = torch.cat(all_pixel_values, dim=0)
        sample[grid_key] = torch.cat(all_image_grid_thw, dim=0)

    return sample


class SampleRepository:
    def __init__(self, args: argparse.Namespace, processor: AutoProcessor):
        self.data_args = _build_data_args(args)
        self.sources = _load_data_sources(args.data_path, args.image_folder)
        self.datasets: List[IterableSupervisedDatasetLVR] = []
        self.offsets: List[int] = [0]
        for source in self.sources:
            dataset = IterableSupervisedDatasetLVR(
                data_path=source["data_path"],
                image_folder=source["image_folder"],
                processor=processor,
                data_args=self.data_args,
                ds_name=source["ds_name"],
                model_id=args.model_id,
                data_rank=0,
                data_world_size=1,
                distributed_mode=False,
                random_seed=None,
                latent_end_token=False,
            )
            self.datasets.append(dataset)
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __len__(self) -> int:
        return self.offsets[-1]

    def resolve(self, global_idx: int) -> Tuple[int, int]:
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(f"Sample index {global_idx} is out of range for length {len(self)}.")
        dataset_idx = bisect.bisect_right(self.offsets, global_idx) - 1
        local_idx = global_idx - self.offsets[dataset_idx]
        return dataset_idx, local_idx

    def get_record_metadata(self, global_idx: int) -> Dict[str, Any]:
        dataset_idx, local_idx = self.resolve(global_idx)
        dataset = self.datasets[dataset_idx]
        raw_record = dataset.raw_data[local_idx]
        return {
            "ds_name": dataset.ds_name,
            "global_idx": global_idx,
            "local_idx": local_idx,
            "question_id": raw_record.get("question_id"),
            "dataset": raw_record.get("dataset"),
            "image": raw_record.get("image"),
        }

    def get_sample(self, global_idx: int) -> Dict[str, Any]:
        dataset_idx, local_idx = self.resolve(global_idx)
        return _materialize_sample(self.datasets[dataset_idx], local_idx)


def _choose_sample_ids(repo: SampleRepository, args: argparse.Namespace) -> List[int]:
    if args.sample_ids_path:
        sample_ids = _normalize_loaded_sample_ids(_load_json(args.sample_ids_path))
    else:
        sample_count = min(args.num_samples, len(repo))
        rng = np.random.default_rng(args.seed)
        sample_ids = rng.choice(len(repo), size=sample_count, replace=False).tolist()

    for sample_id in sample_ids:
        if sample_id < 0 or sample_id >= len(repo):
            raise IndexError(f"Sample id {sample_id} is out of range for repository length {len(repo)}.")

    if args.save_sample_ids_path:
        _dump_json(args.save_sample_ids_path, sample_ids)

    return sample_ids


def _build_single_batch(sample: Dict[str, Any], device: str) -> Dict[str, Any]:
    batch: Dict[str, Any] = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
        "labels": sample["labels"].unsqueeze(0).to(device),
        "lvr_tokens": [token.to(device=device, dtype=torch.long) for token in sample["lvr_tokens"]],
        "lvr_group_to_batch_idx": torch.zeros(len(sample["lvr_tokens"]), dtype=torch.long, device=device),
    }

    stage_sizes = _normalize_stage_sizes(sample.get("lvr_stage_sizes"))
    if stage_sizes is not None:
        batch["lvr_stage_sizes"] = stage_sizes

    if "pixel_values" in sample:
        pixel_values = sample["pixel_values"]
        batch["pixel_values"] = pixel_values.unsqueeze(0).to(device) if pixel_values.dim() == 2 else pixel_values.to(device)
    if "image_grid_thw" in sample:
        batch["image_grid_thw"] = sample["image_grid_thw"].to(device)
    if "second_per_grid_ts" in sample:
        batch["second_per_grid_ts"] = sample["second_per_grid_ts"]

    return batch


def _summarize_stage(hidden_states: torch.Tensor, target_embeds: torch.Tensor) -> Dict[str, Any]:
    cosine = F.cosine_similarity(hidden_states, target_embeds, dim=-1)
    l2_dist = torch.norm(hidden_states - target_embeds, dim=-1)
    mse_per_token = F.mse_loss(hidden_states, target_embeds, reduction="none").mean(dim=-1)
    hidden_norm = torch.norm(hidden_states, dim=-1)
    target_norm = torch.norm(target_embeds, dim=-1)
    norm_ratio = hidden_norm / torch.clamp(target_norm, min=1e-8)
    return {
        "n_tokens": int(hidden_states.shape[0]),
        "mean_cosine_sim": float(cosine.mean().item()),
        "std_cosine_sim": float(cosine.std().item()) if hidden_states.shape[0] > 1 else 0.0,
        "mean_mse": float(mse_per_token.mean().item()),
        "std_mse": float(mse_per_token.std().item()) if hidden_states.shape[0] > 1 else 0.0,
        "mean_l2_dist": float(l2_dist.mean().item()),
        "mean_hidden_norm": float(hidden_norm.mean().item()),
        "mean_target_norm": float(target_norm.mean().item()),
        "mean_norm_ratio": float(norm_ratio.mean().item()),
        "per_token_cosine": cosine.detach().cpu().tolist(),
        "per_token_mse": mse_per_token.detach().cpu().tolist(),
    }


def _split_stage_metrics(
    selected_hidden_states: torch.Tensor,
    selected_lvr_embeds: torch.Tensor,
    stage_sizes: Optional[List[List[int]]],
) -> Dict[str, Dict[str, Any]]:
    if stage_sizes is None:
        return {"target": _summarize_stage(selected_hidden_states, selected_lvr_embeds)}

    context_hidden: List[torch.Tensor] = []
    context_target: List[torch.Tensor] = []
    target_hidden: List[torch.Tensor] = []
    target_target: List[torch.Tensor] = []
    token_offset = 0

    for stage_size in stage_sizes:
        if len(stage_size) == 3:
            n_ctx, _, n_tgt = stage_size
        else:
            n_ctx, n_tgt = stage_size
        if n_ctx > 0:
            context_hidden.append(selected_hidden_states[token_offset : token_offset + n_ctx])
            context_target.append(selected_lvr_embeds[token_offset : token_offset + n_ctx])
        if n_tgt > 0:
            start = token_offset + n_ctx
            target_hidden.append(selected_hidden_states[start : start + n_tgt])
            target_target.append(selected_lvr_embeds[start : start + n_tgt])
        token_offset += n_ctx + n_tgt

    stage_metrics: Dict[str, Dict[str, Any]] = {}
    if context_hidden:
        stage_metrics["context"] = _summarize_stage(torch.cat(context_hidden, dim=0), torch.cat(context_target, dim=0))
    if target_hidden:
        stage_metrics["target"] = _summarize_stage(torch.cat(target_hidden, dim=0), torch.cat(target_target, dim=0))
    return stage_metrics


@torch.no_grad()
def analyze_sample(model: QwenWithLVR, sample: Dict[str, Any], device: str) -> Dict[str, Any]:
    batch = _build_single_batch(sample, device)

    lvr_mask = batch["input_ids"][0] == model.config.lvr_id
    if not lvr_mask.any():
        raise ValueError("Sample has no <|lvr|> tokens after materialization.")

    lvr_positions = torch.where(lvr_mask)[0]
    if torch.any(lvr_positions == 0):
        raise ValueError("Found an <|lvr|> token at sequence position 0.")

    if "pixel_values" not in batch or "image_grid_thw" not in batch:
        raise ValueError("Sample is missing image tensors required for alignment analysis.")

    image_embeds = model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
    image_embeds = torch.cat(image_embeds, dim=0).to(torch.float32)

    flat_indices = torch.cat(batch["lvr_tokens"], dim=0).long()
    if flat_indices.numel() != lvr_positions.numel():
        raise ValueError(
            f"Mismatch between lvr token count ({flat_indices.numel()}) and sequence positions ({lvr_positions.numel()})."
        )
    if flat_indices.numel() == 0:
        raise ValueError("Materialized sample produced zero lvr token indices.")

    clamped_indices = torch.clamp(flat_indices, 0, image_embeds.shape[0] - 1)
    selected_lvr_embeds = image_embeds[clamped_indices]

    outputs = model(**batch, output_hidden_states=True)
    if outputs.hidden_states is None:
        raise ValueError("Model output is missing hidden_states; cannot compute cosine similarity.")

    last_hidden = outputs.hidden_states[-1][0].to(torch.float32)
    prediction_positions = lvr_positions - 1
    selected_hidden_states = last_hidden[prediction_positions]

    stage_sizes = batch.get("lvr_stage_sizes")
    stage_metrics = _split_stage_metrics(selected_hidden_states, selected_lvr_embeds, stage_sizes)

    result: Dict[str, Any] = {
        "n_lvr_tokens": int(lvr_positions.numel()),
        "loss_ce": float(outputs.loss_ce.item()) if outputs.loss_ce is not None else None,
        "loss_lvr": float(outputs.loss_lvr.item()) if outputs.loss_lvr is not None else None,
        "loss_diversity": float(outputs.loss_diversity.item()) if getattr(outputs, "loss_diversity", None) is not None else 0.0,
        "stages": stage_metrics,
        "mean_cosine_sim": stage_metrics.get("target", {}).get("mean_cosine_sim"),
        "mean_target_cosine_sim": stage_metrics.get("target", {}).get("mean_cosine_sim"),
        "mean_context_cosine_sim": stage_metrics.get("context", {}).get("mean_cosine_sim"),
        "mean_target_mse": stage_metrics.get("target", {}).get("mean_mse"),
        "mean_context_mse": stage_metrics.get("context", {}).get("mean_mse"),
        "mean_target_hidden_norm": stage_metrics.get("target", {}).get("mean_hidden_norm"),
        "mean_context_hidden_norm": stage_metrics.get("context", {}).get("mean_hidden_norm"),
        "mean_target_norm_ratio": stage_metrics.get("target", {}).get("mean_norm_ratio"),
        "mean_context_norm_ratio": stage_metrics.get("context", {}).get("mean_norm_ratio"),
    }
    return result


def _aggregate_stage_metric(
    results: Sequence[Dict[str, Any]],
    stage_name: str,
    metric_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    values = [
        result["stages"][stage_name][metric_key]
        for result in results
        if stage_name in result.get("stages", {})
        and result["stages"][stage_name].get(metric_key) is not None
    ]
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def _summarize_results(
    args: argparse.Namespace,
    sample_ids: Sequence[int],
    results: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    loss_ce = [item["loss_ce"] for item in results if item.get("loss_ce") is not None]
    loss_lvr = [item["loss_lvr"] for item in results if item.get("loss_lvr") is not None]
    loss_diversity = [item.get("loss_diversity", 0.0) for item in results]
    mean_target, std_target = _aggregate_stage_metric(results, "target", "mean_cosine_sim")
    mean_context, std_context = _aggregate_stage_metric(results, "context", "mean_cosine_sim")
    mean_target_mse, std_target_mse = _aggregate_stage_metric(results, "target", "mean_mse")
    mean_context_mse, std_context_mse = _aggregate_stage_metric(results, "context", "mean_mse")
    mean_target_hidden_norm, std_target_hidden_norm = _aggregate_stage_metric(results, "target", "mean_hidden_norm")
    mean_context_hidden_norm, std_context_hidden_norm = _aggregate_stage_metric(results, "context", "mean_hidden_norm")
    mean_target_norm_ratio, std_target_norm_ratio = _aggregate_stage_metric(results, "target", "mean_norm_ratio")
    mean_context_norm_ratio, std_context_norm_ratio = _aggregate_stage_metric(results, "context", "mean_norm_ratio")
    valid_ratio = float(len(results) / len(sample_ids)) if sample_ids else 0.0

    return {
        "config": args.config,
        "checkpoint": args.ckpt,
        "n_requested_samples": len(sample_ids),
        "n_valid_samples": len(results),
        "n_failed_samples": len(errors),
        "valid_ratio": valid_ratio,
        "min_valid_ratio": args.min_valid_ratio,
        "mean_loss_ce": float(np.mean(loss_ce)) if loss_ce else None,
        "mean_loss_lvr": float(np.mean(loss_lvr)) if loss_lvr else None,
        "mean_loss_diversity": float(np.mean(loss_diversity)) if loss_diversity else None,
        "mean_cosine_sim": mean_target,
        "mean_target_cosine_sim": mean_target,
        "std_target_cosine_sim": std_target,
        "mean_context_cosine_sim": mean_context,
        "std_context_cosine_sim": std_context,
        "mean_target_mse": mean_target_mse,
        "std_target_mse": std_target_mse,
        "mean_context_mse": mean_context_mse,
        "std_context_mse": std_context_mse,
        "mean_target_hidden_norm": mean_target_hidden_norm,
        "std_target_hidden_norm": std_target_hidden_norm,
        "mean_context_hidden_norm": mean_context_hidden_norm,
        "std_context_hidden_norm": std_context_hidden_norm,
        "mean_target_norm_ratio": mean_target_norm_ratio,
        "std_target_norm_ratio": std_target_norm_ratio,
        "mean_context_norm_ratio": mean_context_norm_ratio,
        "std_context_norm_ratio": std_context_norm_ratio,
        "output_name": _normalize_output_name(args.output_name, args.config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_path", type=str, required=True, help="Meta index or flat shard JSON")
    parser.add_argument("--image_folder", type=str, default=None, help="Used only for flat shard JSON inputs")
    parser.add_argument("--config", type=str, required=True, choices=["baseline", "2stage", "3stage"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./analysis_results")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_ids_path", type=str, default=None)
    parser.add_argument("--save_sample_ids_path", type=str, default=None)
    parser.add_argument("--min_valid_ratio", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _prepare_model(args, device)
    repo = SampleRepository(args, processor)
    sample_ids = _choose_sample_ids(repo, args)

    print(f"Analyzing {len(sample_ids)} samples from a registry of {len(repo)} items...")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for global_idx in tqdm(sample_ids, desc=f"Analyzing {args.config}"):
        metadata = repo.get_record_metadata(global_idx)
        try:
            sample = repo.get_sample(global_idx)
            result = analyze_sample(model, sample, device)
            result.update(metadata)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    **metadata,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = _summarize_results(args, sample_ids, results, errors)
    output_payload = {
        "summary": summary,
        "sample_ids": list(sample_ids),
        "errors": errors,
        "per_sample": results,
    }

    output_path = os.path.join(args.output_dir, _normalize_output_name(args.output_name, args.config))
    _dump_json(output_path, output_payload)
    print(f"Saved to {output_path}")

    if summary["n_valid_samples"] == 0:
        raise SystemExit("No valid samples were analyzed.")
    if summary["valid_ratio"] < args.min_valid_ratio:
        raise SystemExit(
            f"Valid ratio {summary['valid_ratio']:.3f} is below the required threshold {args.min_valid_ratio:.3f}."
        )


if __name__ == "__main__":
    main()
