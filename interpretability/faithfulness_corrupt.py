"""
Faithfulness corruption harness (PRISM Axis 2).

For each (variant, benchmark) cell, runs generation under several corruptions
of the LVR-mode injected hidden state and measures answer accuracy. Δ vs
clean baseline is the faithfulness signal:

  - clean        — no corruption (records baseline accuracy)
  - truncate     — last_position_hidden_state := 0  at every pre_k=True iter
  - noise_0.1    — add N(0, 0.1² I) at every pre_k=True iter
  - noise_0.3    — add N(0, 0.3² I)
  - noise_1.0    — add N(0, 1.0² I)
  - swap         — replace with a random donor sample's injected input at
                   the matching LVR step (stage-tagged for P-LVR)

Implementation per corruption is applied to the *input*
hidden state passed into the forward at each `pre_k=True` iteration, BEFORE
line-533 injection runs. This propagates corruption into the KV cache for
that step, so the entire downstream computation (incl. answer head) reads a
corrupted state.

The clean-baseline pass also records per-iter *injected inputs* per sample;
the donor pool reuses those records (the "separate pre-pass on the
donor set" becomes free).

Run example (one cell, 6 corruptions, ~20 min on H100):

    python interpretability/faithfulness_corrupt.py \\
        --ckpt ${WORKSPACE}/checkpoints/stage1/checkpoint-2500 \\
        --variant lvr_baseline --benchmark vstar --steps 8 \\
        --out-dir ${WORKSPACE}/interpretability_results/faithfulness_20260515
"""

import argparse
import json
import os
import random
import sys
import time
import types
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from transformers.cache_utils import Cache

from qwen_vl_utils import process_vision_info

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import (
    replace_qwen2_5_with_mixed_modality_forward_lvr,
)

from evaluation.evaluation_local import (
    create_messages,
    get_task_instruction,
    load_blink_dataset,
    load_mmvp_dataset,
    load_vstar_dataset,
)
from interpretability.probe_extract import (
    _build_sample_iter,
    _normalize_blink_label,
    _normalize_mmvp_label,
    _parse_answer_letter,
    _proj_h,
)


CORRUPTIONS = ["clean", "truncate", "noise_0.1", "noise_0.3", "noise_1.0", "swap"]


# ---------------------------------------------------------------------------
# Instrumented decoding loops with input-side corruption hook
# ---------------------------------------------------------------------------
#
# The hook applies at every `pre_k=True` iteration BEFORE the forward
# consumes `last_position_hidden_state`. The hook reads its mode from
# `self._fc_state['mode']` and corruption parameters from
# `self._fc_state['params']`. Implementations of the modes:
#
#   - "clean":  no-op
#   - "truncate":  h[b] = 0
#   - "noise_σ": h[b] += σ * randn_like(h[b])
#   - "swap":  h[b] = self._fc_state['donors'][b][step_in_lvr_block]
#
# The hook also records `injected_inputs` (the value of
# last_position_hidden_state at each pre_k=True iter, BEFORE corruption)
# during the clean pass so the donor pool can be built from clean data.


def _corrupt_input(self, last_h, pre_mask, stage_ctx=None, stage_tgt=None):
    """Apply per-sample corruption to `last_h` (B, H) at every batch index
    where `pre_mask[b]` is True. Operates in-place on a clone — original is
    not mutated. Returns the corrupted tensor.

    Per-sample `_fc_state['step_in_block'][b]` is incremented for each
    pre_k=True iter so swap donors are indexed correctly.
    """
    if last_h is None:
        return last_h
    state = self._fc_state
    mode = state["mode"]

    # Always increment per-sample step counters when this iter is pre_k=True
    # (regardless of mode). step_in_ctx / step_in_tgt for P-LVR; step otherwise.
    batch_size = last_h.shape[0]
    for b in range(batch_size):
        if not bool(pre_mask[b]):
            continue
        if stage_tgt is not None and bool(stage_tgt[b]):
            state["step_in_tgt"][b] += 1
        elif stage_ctx is not None and bool(stage_ctx[b]):
            state["step_in_ctx"][b] += 1
        else:
            state["step"][b] += 1

    if mode == "clean":
        # Record per-sample injected input for donor pool construction.
        rec = state.get("injected")
        if rec is not None:
            for b in range(batch_size):
                if not bool(pre_mask[b]):
                    continue
                vec = last_h[b].detach().to(torch.float16).cpu().clone()
                if stage_tgt is not None and bool(stage_tgt[b]):
                    rec[b]["tgt"].append(vec)
                elif stage_ctx is not None and bool(stage_ctx[b]):
                    rec[b]["ctx"].append(vec)
                else:
                    rec[b]["single"].append(vec)
        return last_h

    h = last_h.clone()
    for b in range(batch_size):
        if not bool(pre_mask[b]):
            continue
        if mode == "truncate":
            h[b] = 0.0
        elif mode.startswith("noise_"):
            sigma = state["sigma"]
            h[b] = h[b] + sigma * torch.randn_like(h[b])
        elif mode == "swap":
            donors = state["donors"]
            # Pick this sample's donor; index by stage step.
            donor = donors[b]
            if stage_tgt is not None and bool(stage_tgt[b]):
                idx = state["step_in_tgt"][b] - 1   # we just incremented
                slots = donor["tgt"]
            elif stage_ctx is not None and bool(stage_ctx[b]):
                idx = state["step_in_ctx"][b] - 1
                slots = donor["ctx"]
            else:
                idx = state["step"][b] - 1
                slots = donor["single"]
            if slots is None or len(slots) == 0:
                # Donor has no records for this stage; fall back to noise.
                h[b] = h[b] + 0.3 * torch.randn_like(h[b])
            else:
                k = min(idx, len(slots) - 1)
                h[b] = slots[k].to(h.dtype).to(h.device)
        else:
            raise ValueError(f"Unknown corruption mode: {mode}")
    return h


def _make_patched_single_stage_loop():
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    def _patched(
        self, input_ids, logits_processor, stopping_criteria, generation_config,
        synced_gpus, streamer, lvr_steps, **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        do_sample = generation_config.do_sample
        scores = ()

        batch_size, cur_len = input_ids.shape
        prompt_len = cur_len
        this_peer_finished = False
        unfinished = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        try:
            model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        except Exception:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = (
                model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            )
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        lvr_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        last_position_hidden_state = None
        lvr_steps_orig = torch.tensor(lvr_steps, dtype=torch.int, device=input_ids.device)
        lvr_remaining_steps = lvr_steps_orig.clone()

        # Reset per-sample step counters for this generate() call.
        self._fc_state["step"] = [0] * batch_size
        self._fc_state["step_in_ctx"] = [0] * batch_size
        self._fc_state["step_in_tgt"] = [0] * batch_size

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs["lvr_mode_switch"] = lvr_mode_switch

            # APPLY CORRUPTION HOOK before the forward consumes the parameter.
            corrupted = _corrupt_input(self, last_position_hidden_state, lvr_mode_switch)
            model_inputs["last_position_hidden_state"] = corrupted

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if has_eos:
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

            last_tokens = input_ids[:, -1]
            lvr_start_switch = (last_tokens == self.config.lvr_start_id).to(device=input_ids.device)
            for _start_attr in ("ctx_start_id", "tgt_start_id"):
                _sid = getattr(self.config, _start_attr, None)
                if _sid is not None:
                    lvr_start_switch = lvr_start_switch | (last_tokens == _sid)

            new_mode = lvr_mode_switch | lvr_start_switch
            just_entered = (~lvr_mode_switch) & new_mode
            lvr_remaining_steps = torch.where(just_entered, lvr_steps_orig, lvr_remaining_steps)
            lvr_remaining_steps = lvr_remaining_steps - lvr_mode_switch.long()
            lvr_mode_switch = new_mode & (lvr_remaining_steps > 0)

            last_position_hidden_state = _proj_h(self, outputs.last_position_hidden_state)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished = lvr_mode_switch | (unfinished & ~stopping_criteria(input_ids, scores))
            this_peer_finished = unfinished.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()
        self._fc_state["prompt_len"] = prompt_len
        self._fc_state["sequences"] = input_ids.detach().cpu().clone()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=None, logits=None, attentions=None, hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids
    return _patched


def _make_patched_plvr_loop():
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    def _patched(
        self, input_ids, logits_processor, stopping_criteria, generation_config,
        synced_gpus, streamer, lvr_steps, plvr_target_only=False, **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        do_sample = generation_config.do_sample
        scores = ()

        batch_size, cur_len = input_ids.shape
        prompt_len = cur_len
        this_peer_finished = False
        unfinished = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        try:
            model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        except Exception:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = (
                model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            )
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        gc_was_enabled = getattr(self, "is_gradient_checkpointing", False)
        if gc_was_enabled:
            self.gradient_checkpointing_disable()

        lvr_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        last_position_hidden_state = None
        lvr_steps_orig = torch.tensor(lvr_steps, dtype=torch.int, device=input_ids.device)
        ctx_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        tgt_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        ctx_steps_orig = lvr_steps_orig * 2
        ctx_remaining_steps = ctx_steps_orig.clone()
        tgt_remaining_steps = lvr_steps_orig.clone()

        self._fc_state["step"] = [0] * batch_size
        self._fc_state["step_in_ctx"] = [0] * batch_size
        self._fc_state["step_in_tgt"] = [0] * batch_size

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs["lvr_mode_switch"] = lvr_mode_switch

            # Note: ctx/tgt pre-update masks are the BEFORE-update values
            # passed alongside lvr_mode_switch.
            corrupted = _corrupt_input(
                self, last_position_hidden_state, lvr_mode_switch,
                stage_ctx=ctx_mode_switch, stage_tgt=tgt_mode_switch,
            )
            model_inputs["last_position_hidden_state"] = corrupted

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if has_eos:
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

            last_tokens = input_ids[:, -1]
            ctx_start_switch = (last_tokens == self.config.ctx_start_id).to(device=input_ids.device)
            if plvr_target_only:
                ctx_start_switch = torch.zeros_like(ctx_start_switch)
            new_ctx = ctx_mode_switch | ctx_start_switch
            just_entered_ctx = (~ctx_mode_switch) & new_ctx
            ctx_remaining_steps = torch.where(just_entered_ctx, ctx_steps_orig, ctx_remaining_steps)
            ctx_remaining_steps = ctx_remaining_steps - ctx_mode_switch.long()
            ctx_mode_switch = new_ctx & (ctx_remaining_steps > 0)

            tgt_start_switch = (last_tokens == self.config.tgt_start_id).to(device=input_ids.device)
            new_tgt = tgt_mode_switch | tgt_start_switch
            just_entered_tgt = (~tgt_mode_switch) & new_tgt
            tgt_remaining_steps = torch.where(just_entered_tgt, lvr_steps_orig, tgt_remaining_steps)
            tgt_remaining_steps = tgt_remaining_steps - tgt_mode_switch.long()
            tgt_mode_switch = new_tgt & (tgt_remaining_steps > 0)

            lvr_mode_switch = ctx_mode_switch | tgt_mode_switch

            last_position_hidden_state = _proj_h(self, outputs.last_position_hidden_state)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished = lvr_mode_switch | (unfinished & ~stopping_criteria(input_ids, scores))
            this_peer_finished = unfinished.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()
        if gc_was_enabled:
            self.gradient_checkpointing_enable()
        self._fc_state["prompt_len"] = prompt_len
        self._fc_state["sequences"] = input_ids.detach().cpu().clone()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=None, logits=None, attentions=None, hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids
    return _patched


def install_corruption_instrumentation(model):
    patched_single = _make_patched_single_stage_loop()
    patched_plvr = _make_patched_plvr_loop()
    model._lvr_deocding_by_steps = types.MethodType(patched_single, model)
    model._lvr_decoding_by_steps_plvr = types.MethodType(patched_plvr, model)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_model_and_processor(ckpt):
    config = AutoConfig.from_pretrained(ckpt)
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=getattr(config, "lvr_head", False),
    )
    model = QwenWithLVR.from_pretrained(
        ckpt, config=config, trust_remote_code=True, torch_dtype="auto",
        attn_implementation="flash_attention_2", device_map="auto",
    )
    model.eval()
    return model, AutoProcessor.from_pretrained(ckpt)


BENCH_LOADERS = {
    "vstar": load_vstar_dataset,
    "MMVP": load_mmvp_dataset,
    "blink": load_blink_dataset,
}


def _run_batch(model, processor, items, steps, mode, donors=None, sigma=None, injected_rec=None):
    """Run a batch with the requested corruption mode. Returns predicted
    letters (one per sample) for accuracy computation.
    """
    messages = [create_messages(img, txt) for img, txt in items]
    text_formatted = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text_formatted, images=image_inputs, videos=video_inputs,
        padding=True, padding_side="left", return_tensors="pt",
    ).to("cuda")
    batch_size = len(items)

    model._fc_state = {
        "mode": mode,
        "sigma": sigma,
        "donors": donors,
        "injected": injected_rec,         # list of {"single": [...], "ctx": [...], "tgt": [...]} per batch index, or None
        "step": [0] * batch_size,
        "step_in_ctx": [0] * batch_size,
        "step_in_tgt": [0] * batch_size,
        "prompt_len": None,
        "sequences": None,
    }
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=512, decoding_strategy="steps",
            lvr_steps=[steps] * batch_size, plvr_target_only=False,
        )

    if hasattr(generated, "sequences"):
        sequences = generated.sequences.detach().cpu()
    else:
        sequences = generated.detach().cpu()
    prompt_len = inputs.input_ids.shape[1]
    preds = []
    for b in range(batch_size):
        tail = sequences[b, prompt_len:]
        preds.append(_parse_answer_letter(processor.tokenizer, tail))
    return preds


def _build_donor_pool_from_records(injected_rec, pool_size):
    """From the per-sample injected-records dict list, sample `pool_size`
    donors. Each donor is a dict with 'single' / 'ctx' / 'tgt' slot lists.
    """
    donors_all = [r for r in injected_rec if (
        len(r["single"]) > 0 or len(r["ctx"]) > 0 or len(r["tgt"]) > 0
    )]
    if not donors_all:
        return []
    if len(donors_all) <= pool_size:
        return donors_all
    return random.sample(donors_all, pool_size)


def _pick_donors_for_batch(donor_pool, batch_size):
    """For each sample in the batch, pick a random donor from the pool."""
    return [random.choice(donor_pool) for _ in range(batch_size)]


def _accuracy(preds: List[Optional[str]], labels: List[str]) -> Tuple[float, int]:
    correct = 0
    n = 0
    for p, l in zip(preds, labels):
        if p is None:
            n += 1
            continue
        if p.upper() == (l or "").upper():
            correct += 1
        n += 1
    return correct / n if n else 0.0, n


def run_cell(args):
    print(f"[fc] ckpt={args.ckpt}")
    print(f"[fc] variant={args.variant} benchmark={args.benchmark} steps={args.steps}")
    model, processor = load_model_and_processor(args.ckpt)
    install_corruption_instrumentation(model)
    is_plvr = bool(getattr(model.config, "plvr_mode", False))
    print(f"[fc] is_plvr={is_plvr}")

    dataset, image_dir, _, ds_name = BENCH_LOADERS[args.benchmark](
        getattr(model.config, "lvr_head", False), "faith", "steps",
    )
    print(f"[fc] {args.benchmark}: {len(dataset)} samples")

    bench_for_iter = "MMVP" if args.benchmark.upper() == "MMVP" else args.benchmark
    sample_iter = list(_build_sample_iter(dataset, bench_for_iter, image_dir))
    if args.limit is not None and args.limit > 0:
        sample_iter = sample_iter[: args.limit]
        print(f"[fc] limited to {len(sample_iter)} samples")

    bs = max(1, int(args.batch_size))

    # Per-sample bookkeeping.
    all_labels = []
    for tup in sample_iter:
        _, _, _, label, _ = tup
        all_labels.append(_normalize_blink_label(label) if args.benchmark == "blink" else label)

    results = {}
    # All injected-input records from the CLEAN pass (for donor pool).
    injected_all: List[Dict] = []

    # Iterate corruptions in order: clean first (so we have donor data for swap).
    for mode in CORRUPTIONS:
        if mode.startswith("noise_"):
            sigma = float(mode.split("_")[1])
        else:
            sigma = None

        # Donor pool needed for swap mode.
        donor_pool = None
        if mode == "swap":
            donor_pool = _build_donor_pool_from_records(injected_all, args.donor_pool_size)
            print(f"[fc] swap: donor pool size = {len(donor_pool)}")
            if not donor_pool:
                print("[fc] WARN: no donors available; skipping swap")
                continue

        all_preds: List[Optional[str]] = []
        pbar = tqdm(range(0, len(sample_iter), bs), desc=f"{args.variant}/{args.benchmark}/{mode}")
        for start in pbar:
            chunk = sample_iter[start:start + bs]
            items = [(img, text) for (_, img, text, _, _) in chunk]
            # Per-batch injected-record holder (clean only).
            if mode == "clean":
                injected_rec = [
                    {"single": [], "ctx": [], "tgt": []} for _ in range(len(chunk))
                ]
            else:
                injected_rec = None

            # For swap: pick a donor for each sample in batch.
            if mode == "swap" and donor_pool:
                donors = _pick_donors_for_batch(donor_pool, len(chunk))
            else:
                donors = None

            try:
                preds = _run_batch(
                    model, processor, items, args.steps, mode,
                    donors=donors, sigma=sigma, injected_rec=injected_rec,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[fc] OOM at start={start}; falling back per-sample")
                preds = []
                for it in items:
                    try:
                        if mode == "clean":
                            sub_rec = [{"single": [], "ctx": [], "tgt": []}]
                        else:
                            sub_rec = None
                        sub_donors = [donors[0]] if donors else None
                        p = _run_batch(
                            model, processor, [it], args.steps, mode,
                            donors=sub_donors, sigma=sigma, injected_rec=sub_rec,
                        )[0]
                        if mode == "clean":
                            injected_rec.append(sub_rec[0]) if False else None
                            injected_all.append(sub_rec[0])
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        p = None
                    preds.append(p)
            all_preds.extend(preds)

            # Stash this batch's injected records.
            if mode == "clean" and injected_rec is not None:
                injected_all.extend(injected_rec)

        acc, n = _accuracy(all_preds, all_labels[:len(all_preds)])
        results[mode] = {"accuracy": acc, "n": n}
        print(f"[fc] {mode}: acc={acc*100:.2f}% (n={n})")

    # Compute Δ vs clean.
    clean_acc = results.get("clean", {}).get("accuracy", None)
    if clean_acc is not None:
        for mode, r in results.items():
            r["delta_vs_clean"] = r["accuracy"] - clean_acc

    out_payload = {
        "variant": args.variant,
        "benchmark": args.benchmark,
        "checkpoint": args.ckpt,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": bs,
        "is_plvr": is_plvr,
        "donor_pool_size": args.donor_pool_size,
        "n_samples": len(sample_iter),
        "results": results,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_name = f"faith_{args.variant}_{args.benchmark}_seed{args.seed}.json"
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[fc] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--benchmark", required=True, choices=list(BENCH_LOADERS.keys()))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--donor-pool-size", type=int, default=64,
                    help="Donors sampled from the clean pass for swap corruption.")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    run_cell(args)


if __name__ == "__main__":
    main()
