"""
Linear-probe hidden-state extraction harness (PRISM Axis 1).

For each (variant, benchmark) cell, this script runs the model's normal
`decoding_strategy="steps"` generation on every sample, but additionally
instruments the LVR-mode generation loops to record per-iteration hidden
states. After generation, five-to-seven hidden-state probes per sample
are saved to a `.pt` cache for downstream `probe_train.py`.

Run example (one job):

    python interpretability/probe_extract.py \
        --ckpt ${WORKSPACE}/checkpoints/stage1/checkpoint-2500 \
        --variant lvr_baseline \
        --benchmark vstar \
        --steps 8 \
        --out-dir ${WORKSPACE}/interpretability_results/probes_20260514

Paper anchors:
- (probe positions a / b / c)
- (dual-loop instrumentation, raw vs proj, answer-start anchor for (a),
        stage-aware (b)/(c))
- (output schema)
- (A1' question-mean / A1'' visual-mean baselines from prefill)
"""

import argparse
import json
import os
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

from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    LVR_END_TOKEN,
    LVR_START_TOKEN,
    PLVR_CTX_END_TOKEN,
    PLVR_CTX_START_TOKEN,
    PLVR_FREE_END_TOKEN,
    PLVR_FREE_START_TOKEN,
    PLVR_TGT_END_TOKEN,
    PLVR_TGT_START_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import (
    replace_qwen2_5_with_mixed_modality_forward_lvr,
)

# Reuse the eval dataset loaders / task instruction unchanged.
from evaluation.evaluation_local import (
    create_messages,
    get_task_instruction,
    load_blink_dataset,
    load_mmvp_dataset,
    load_vstar_dataset,
)


# ---------------------------------------------------------------------------
# Instrumented decoding loops
# ---------------------------------------------------------------------------
#
# These are line-for-line copies of `_lvr_deocding_by_steps` and
# `_lvr_decoding_by_steps_plvr` in `src/model/qwen_lvr_model.py`, with three
# additions:
#   1. `output_hidden_states=True` is forced on the prefill iteration so that
#      `outputs.hidden_states[-1]` is available for the A1' / A1'' baselines
#     . It is NOT set on subsequent iterations — that would
#      blow up memory.
#   2. Per iteration k, we record the *pre-update* `lvr_mode_switch`
#      ("records are keyed by `pre_k`"), the raw hidden state
#      `outputs.last_position_hidden_state`, and the *post-update* switch.
#   3. For P-LVR, the pre-update `ctx_mode_switch` and `tgt_mode_switch` are
#      recorded so we can stage-tag each `pre_k=True` iteration.
#
# Records are appended to `self._probe_records`, which is a fresh dict
# installed by the caller before each sample's `model.generate(...)`.


def _proj_h(model: nn.Module, raw_h: torch.Tensor) -> torch.Tensor:
    """Return the feedback variable. The five paper variants have no
    learned projection between the LVR-position hidden state and the next
    input embedding, so this is the identity.
    """
    return raw_h


def _make_patched_single_stage_loop():
    # Local imports to keep module top-level lean.
    from transformers.generation.utils import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerateNonBeamOutput,
    )

    def _patched(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        lvr_steps,
        **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        # Force hidden states for prefill so A1' / A1'' can be computed.
        output_hidden_states_cfg = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states_cfg) else None

        batch_size, cur_len = input_ids.shape
        prompt_len = cur_len
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
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

        records = self._probe_records["iters"]

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if output_attentions:
                model_inputs["output_attentions"] = output_attentions
            # Hidden states: only on prefill (saves memory).
            model_inputs["output_hidden_states"] = bool(is_prefill)
            model_inputs["lvr_mode_switch"] = lvr_mode_switch
            model_inputs["last_position_hidden_state"] = last_position_hidden_state

            pre_k = lvr_mode_switch.detach().cpu().clone()

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                # Capture prefill hidden states (last layer) and input_ids for
                # A1' / A1'' baselines.
                if outputs.hidden_states is not None:
                    self._probe_records["prefill_hidden_last_layer"] = (
                        outputs.hidden_states[-1].detach().to(torch.float16).cpu().clone()
                    )
                else:
                    self._probe_records["prefill_hidden_last_layer"] = None
                self._probe_records["prefill_input_ids"] = input_ids.detach().cpu().clone()
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            last_tokens = input_ids[:, -1]
            lvr_start_switch = (last_tokens == self.config.lvr_start_id).to(device=input_ids.device)
            for _stage_start_attr in ("ctx_start_id", "tgt_start_id"):
                _sid = getattr(self.config, _stage_start_attr, None)
                if _sid is not None:
                    lvr_start_switch = lvr_start_switch | (last_tokens == _sid)

            new_mode_switch = lvr_mode_switch | lvr_start_switch
            just_entered = (~lvr_mode_switch) & new_mode_switch
            lvr_remaining_steps = torch.where(just_entered, lvr_steps_orig, lvr_remaining_steps)
            lvr_remaining_steps = lvr_remaining_steps - lvr_mode_switch.long()
            lvr_mode_switch = new_mode_switch & (lvr_remaining_steps > 0)

            # Record AFTER post-update so we have both pre and post.
            raw_h = outputs.last_position_hidden_state  # (B, H)
            with torch.no_grad():
                proj_h = _proj_h(self, raw_h)
            records.append({
                "pre": pre_k,                                                        # (B,) bool, CPU
                "post": lvr_mode_switch.detach().cpu().clone(),                      # (B,) bool, CPU
                "raw_h": raw_h.detach().to(torch.float16).cpu().clone(),             # (B, H) fp16 CPU
                "proj_h": proj_h.detach().to(torch.float16).cpu().clone(),           # (B, H) fp16 CPU
                "next_tokens": next_tokens.detach().cpu().clone(),                   # (B,)
                # Stage tags absent in single-stage loop — fill with all-False so
                # downstream stage-aware code can be uniform.
                "ctx_pre": torch.zeros_like(pre_k),
                "tgt_pre": torch.zeros_like(pre_k),
            })

            # Feed back the raw hidden state; matches qwen_lvr_model.py.
            last_position_hidden_state = proj_h
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = (
                lvr_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        self._probe_records["prompt_len"] = prompt_len
        self._probe_records["sequences"] = input_ids.detach().cpu().clone()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                # Path not exercised here, but keep parity.
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=None,
                    encoder_hidden_states=None,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    return _patched


def _make_patched_plvr_loop():
    from transformers.generation.utils import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
    )

    def _patched(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        lvr_steps,
        plvr_target_only=False,
        **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states_cfg = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states_cfg) else None

        batch_size, cur_len = input_ids.shape
        prompt_len = cur_len
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
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

        records = self._probe_records["iters"]

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if output_attentions:
                model_inputs["output_attentions"] = output_attentions
            model_inputs["output_hidden_states"] = bool(is_prefill)
            model_inputs["lvr_mode_switch"] = lvr_mode_switch
            model_inputs["last_position_hidden_state"] = last_position_hidden_state

            pre_k = lvr_mode_switch.detach().cpu().clone()
            ctx_pre = ctx_mode_switch.detach().cpu().clone()
            tgt_pre = tgt_mode_switch.detach().cpu().clone()

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                if outputs.hidden_states is not None:
                    self._probe_records["prefill_hidden_last_layer"] = (
                        outputs.hidden_states[-1].detach().to(torch.float16).cpu().clone()
                    )
                else:
                    self._probe_records["prefill_hidden_last_layer"] = None
                self._probe_records["prefill_input_ids"] = input_ids.detach().cpu().clone()
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            last_tokens = input_ids[:, -1]

            # ctx-stage update.
            ctx_start_switch = (last_tokens == self.config.ctx_start_id).to(device=input_ids.device)
            if plvr_target_only:
                ctx_start_switch = torch.zeros_like(ctx_start_switch)
            new_ctx_mode = ctx_mode_switch | ctx_start_switch
            just_entered_ctx = (~ctx_mode_switch) & new_ctx_mode
            ctx_remaining_steps = torch.where(just_entered_ctx, ctx_steps_orig, ctx_remaining_steps)
            ctx_remaining_steps = ctx_remaining_steps - ctx_mode_switch.long()
            ctx_mode_switch = new_ctx_mode & (ctx_remaining_steps > 0)

            # tgt-stage update.
            tgt_start_switch = (last_tokens == self.config.tgt_start_id).to(device=input_ids.device)
            new_tgt_mode = tgt_mode_switch | tgt_start_switch
            just_entered_tgt = (~tgt_mode_switch) & new_tgt_mode
            tgt_remaining_steps = torch.where(just_entered_tgt, lvr_steps_orig, tgt_remaining_steps)
            tgt_remaining_steps = tgt_remaining_steps - tgt_mode_switch.long()
            tgt_mode_switch = new_tgt_mode & (tgt_remaining_steps > 0)

            lvr_mode_switch = ctx_mode_switch | tgt_mode_switch

            raw_h = outputs.last_position_hidden_state
            with torch.no_grad():
                proj_h = _proj_h(self, raw_h)
            records.append({
                "pre": pre_k,
                "post": lvr_mode_switch.detach().cpu().clone(),
                "raw_h": raw_h.detach().to(torch.float16).cpu().clone(),
                "proj_h": proj_h.detach().to(torch.float16).cpu().clone(),
                "next_tokens": next_tokens.detach().cpu().clone(),
                "ctx_pre": ctx_pre,
                "tgt_pre": tgt_pre,
            })

            # Feed back the raw hidden state; matches qwen_lvr_model.py.
            last_position_hidden_state = proj_h
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = (
                lvr_mode_switch | (unfinished_sequences & ~stopping_criteria(input_ids, scores))
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()
        if gc_was_enabled:
            self.gradient_checkpointing_enable()

        self._probe_records["prompt_len"] = prompt_len
        self._probe_records["sequences"] = input_ids.detach().cpu().clone()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=None,
                    encoder_hidden_states=None,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    return _patched


def install_probe_instrumentation(model):
    """Replace the two LVR-mode decoding loops on the loaded model instance
    with instrumented copies that record per-iteration hidden states.

    The originals on the class are left untouched.
    """
    patched_single = _make_patched_single_stage_loop()
    patched_plvr = _make_patched_plvr_loop()
    model._lvr_deocding_by_steps = types.MethodType(patched_single, model)
    model._lvr_decoding_by_steps_plvr = types.MethodType(patched_plvr, model)


# ---------------------------------------------------------------------------
# Per-sample probe extraction
# ---------------------------------------------------------------------------


def _decode_token_strs(tokenizer, token_ids: torch.Tensor) -> List[str]:
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids]


def _find_first_answer_token_idx(tokenizer, tail_token_ids: torch.Tensor) -> Optional[int]:
    """Return the index in `tail_token_ids` of the first token whose decoded
    text appears *after* a complete `<answer>` substring.

    Returns None if `<answer>` is not present in the generated tail.
    """
    text = ""
    answer_marker = "<answer>"
    found_at = None
    for i in range(tail_token_ids.shape[0]):
        text += tokenizer.decode([int(tail_token_ids[i])], skip_special_tokens=False)
        if found_at is None and answer_marker in text:
            found_at = i  # this token completed `<answer>`
            # The first answer-text token is the NEXT one.
            if i + 1 < tail_token_ids.shape[0]:
                return i + 1
            return i  # degenerate: model emitted `<answer>` and stopped.
    return None


def _parse_answer_letter(tokenizer, tail_token_ids: torch.Tensor) -> Optional[str]:
    text = tokenizer.decode(tail_token_ids, skip_special_tokens=False)
    if "<answer>" not in text:
        return None
    snippet = text.split("<answer>")[-1].split("</answer>")[0].strip()
    if not snippet:
        return None
    # Mirror evaluation_local.accuracy_reward parsing.
    if " " in snippet:
        snippet = snippet.split(" ")[0]
    if len(snippet) > 1:
        snippet = snippet[0]
    return snippet


def _build_special_ids(tokenizer):
    def _id(tok):
        i = tokenizer.convert_tokens_to_ids(tok)
        return i if i is not None and i != tokenizer.unk_token_id else None

    return {
        "im_start": _id(DEFAULT_IM_START_TOKEN),
        "im_end": _id(DEFAULT_IM_END_TOKEN),
        "image_pad": _id(DEFAULT_IMAGE_TOKEN),
        "vision_start": _id(VISION_START_TOKEN),
        "vision_end": _id(VISION_END_TOKEN),
    }


def _question_and_visual_means(
    prefill_input_ids: torch.Tensor,  # (1, prompt_len)
    prefill_hidden: torch.Tensor,     # (1, prompt_len, H), fp16 CPU
    special_ids: Dict[str, Optional[int]],
    tokenizer,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute the A1' (question-mean) and A1'' (visual-mean) hidden states
    from the prefill pass..

    - Question tokens: the user-turn span between `<|im_start|>user` and the
      next `<|im_end|>`, with `<|vision_start|>...<|vision_end|>` segments
      (image tokens) excluded.
    - Visual tokens: positions where `input_ids == <|image_pad|>` (covers
      multi-image BLINK samples — pool over all image-pad positions).
    """
    if prefill_hidden is None or prefill_input_ids is None:
        return None, None
    ids = prefill_input_ids[0]  # (prompt_len,)
    h = prefill_hidden[0].to(torch.float32)  # (prompt_len, H)

    # Visual mean: simply all `image_pad` positions.
    visual_vec = None
    pad_id = special_ids.get("image_pad")
    if pad_id is not None:
        mask = ids == pad_id
        if mask.any():
            visual_vec = h[mask].mean(dim=0).cpu().numpy().astype(np.float16)

    # Question mean: scan for the user turn.
    question_vec = None
    im_start_id = special_ids.get("im_start")
    im_end_id = special_ids.get("im_end")
    vis_start = special_ids.get("vision_start")
    vis_end = special_ids.get("vision_end")
    if im_start_id is not None and im_end_id is not None:
        # Find all <|im_start|> positions; the user turn is the one whose next
        # tokens decode to "user".
        starts = (ids == im_start_id).nonzero(as_tuple=False).flatten().tolist()
        user_start = None
        for s in starts:
            # decode the 1-2 tokens after to identify the role.
            if s + 1 >= ids.shape[0]:
                continue
            role_str = tokenizer.decode([int(ids[s + 1])], skip_special_tokens=False).strip()
            if role_str.startswith("user"):
                user_start = s
                break
        if user_start is not None:
            # Find the next <|im_end|> after user_start.
            ends = (ids == im_end_id).nonzero(as_tuple=False).flatten().tolist()
            user_end = next((e for e in ends if e > user_start), None)
            if user_end is not None:
                span_mask = torch.zeros_like(ids, dtype=torch.bool)
                span_mask[user_start + 1:user_end] = True
                # Exclude vision_start..vision_end (image embeds, not text).
                if vis_start is not None and vis_end is not None:
                    in_vision = False
                    for p in range(user_start + 1, user_end):
                        tid = int(ids[p])
                        if tid == vis_start:
                            in_vision = True
                            span_mask[p] = False
                            continue
                        if tid == vis_end:
                            in_vision = False
                            span_mask[p] = False
                            continue
                        if in_vision:
                            span_mask[p] = False
                # Also exclude any leftover image_pad (defensive).
                if pad_id is not None:
                    span_mask = span_mask & (ids != pad_id)
                if span_mask.any():
                    question_vec = h[span_mask].mean(dim=0).cpu().numpy().astype(np.float16)
    return question_vec, visual_vec


def _extract_probes_from_records(
    records: dict,
    tokenizer,
    is_plvr_loop: bool,
    special_ids: Dict[str, Optional[int]],
    batch_index: int = 0,
) -> Optional[Dict]:
    """Per-sample post-processing.

    Returns a dict with the hidden_a/b/c/b_ctx/c_ctx/a1_* numpy arrays and
    bookkeeping fields, or None on extraction failure.
    """
    b = batch_index
    iters = records["iters"]
    if not iters:
        return None

    pre = torch.stack([it["pre"][b] for it in iters])           # (K,) bool
    post = torch.stack([it["post"][b] for it in iters])
    raw_h = torch.stack([it["raw_h"][b] for it in iters])       # (K, H) fp16
    proj_h = torch.stack([it["proj_h"][b] for it in iters])
    ctx_pre = torch.stack([it["ctx_pre"][b] for it in iters])
    tgt_pre = torch.stack([it["tgt_pre"][b] for it in iters])
    next_tokens = torch.stack([it["next_tokens"][b] for it in iters])  # (K,)

    # Compute hidden_a: iteration that produces the first answer-text token.
    sequences = records["sequences"]                                       # (B, prompt+tail)
    prompt_len = records["prompt_len"]
    tail = sequences[b, prompt_len:]
    # `next_tokens[k]` was concatenated as `tail[k]`. So if first answer-text
    # token is at tail index p, hidden_a comes from raw_h[p].
    ans_p = _find_first_answer_token_idx(tokenizer, tail)
    predicted_letter = _parse_answer_letter(tokenizer, tail)
    if ans_p is None:
        # Fallback per first pre_k=False after the latent block exits
        # (for single-stage); for P-LVR, iteration immediately following the
        # LAST latent block exit.
        if is_plvr_loop:
            true_idx = pre.nonzero(as_tuple=False).flatten()
            if true_idx.numel() == 0:
                return None
            last_true = int(true_idx[-1].item())
            ans_p = min(last_true + 1, pre.shape[0] - 1)
        else:
            true_idx = pre.nonzero(as_tuple=False).flatten()
            if true_idx.numel() == 0:
                ans_p = 0
            else:
                last_true = int(true_idx[-1].item())
                ans_p = min(last_true + 1, pre.shape[0] - 1)
    if ans_p >= raw_h.shape[0]:
        ans_p = raw_h.shape[0] - 1
    hidden_a = raw_h[ans_p].cpu().numpy().astype(np.float16)

    # Compute hidden_b/c — stage-aware.
    # Latents = iterations with pre_k=True.
    lvr_idx = pre.nonzero(as_tuple=False).flatten()
    if lvr_idx.numel() == 0:
        # No LVR-mode iterations recorded — degenerate; skip sample.
        return None

    if is_plvr_loop:
        ctx_idx = (ctx_pre & pre).nonzero(as_tuple=False).flatten()
        tgt_idx = (tgt_pre & pre).nonzero(as_tuple=False).flatten()
        stages_present = []
        if ctx_idx.numel() > 0:
            stages_present.append("ctx")
        if tgt_idx.numel() > 0:
            stages_present.append("tgt")
        # latest stage = tgt if it fired, else ctx.
        if tgt_idx.numel() > 0:
            latest_idx = tgt_idx
        elif ctx_idx.numel() > 0:
            latest_idx = ctx_idx
        else:
            return None
        hidden_b = proj_h[int(latest_idx[-1].item())].cpu().numpy().astype(np.float16)
        hidden_c = proj_h[latest_idx].to(torch.float32).mean(dim=0).cpu().numpy().astype(np.float16)
        if ctx_idx.numel() > 0:
            hidden_b_ctx = proj_h[int(ctx_idx[-1].item())].cpu().numpy().astype(np.float16)
            hidden_c_ctx = proj_h[ctx_idx].to(torch.float32).mean(dim=0).cpu().numpy().astype(np.float16)
        else:
            hidden_b_ctx = None
            hidden_c_ctx = None
    else:
        stages_present = ["single"]
        hidden_b = proj_h[int(lvr_idx[-1].item())].cpu().numpy().astype(np.float16)
        hidden_c = proj_h[lvr_idx].to(torch.float32).mean(dim=0).cpu().numpy().astype(np.float16)
        hidden_b_ctx = None
        hidden_c_ctx = None

    # A1' / A1''.
    prefill_input_ids = records.get("prefill_input_ids")
    prefill_hidden = records.get("prefill_hidden_last_layer")
    if prefill_input_ids is not None and prefill_hidden is not None:
        # Slice the batch index for this sample.
        pi = prefill_input_ids[b:b + 1]
        ph = prefill_hidden[b:b + 1]
    else:
        pi, ph = None, None
    q_mean, v_mean = _question_and_visual_means(
        pi, ph, special_ids, tokenizer,
    )

    return {
        "hidden_a": hidden_a,
        "hidden_b": hidden_b,
        "hidden_c": hidden_c,
        "hidden_b_ctx": hidden_b_ctx,
        "hidden_c_ctx": hidden_c_ctx,
        "hidden_a1_question_mean": q_mean,
        "hidden_a1_visual_mean": v_mean,
        "lvr_stages_present": stages_present,
        "n_lvr_steps": int(lvr_idx.numel()),
        "n_ctx_steps": int(ctx_pre.sum().item()) if is_plvr_loop else 0,
        "n_tgt_steps": int(tgt_pre.sum().item()) if is_plvr_loop else 0,
        "answer_anchor_iter": int(ans_p),
        "answer_anchor_found": bool(_find_first_answer_token_idx(tokenizer, tail) is not None),
        "predicted_letter": predicted_letter,
        "tail_token_count": int(tail.shape[0]),
    }


# ---------------------------------------------------------------------------
# Model loading and benchmark dispatch
# ---------------------------------------------------------------------------


def load_model_and_processor(ckpt_pth: str):
    """Mirrors evaluation_local.load_model_and_processor."""
    config = AutoConfig.from_pretrained(ckpt_pth)
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=getattr(config, "lvr_head", False)
    )
    model = QwenWithLVR.from_pretrained(
        ckpt_pth,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(ckpt_pth)
    return model, processor


BENCH_LOADERS = {
    "vstar": load_vstar_dataset,
    "MMVP": load_mmvp_dataset,
    "blink": load_blink_dataset,
}


def _normalize_blink_label(label: str) -> str:
    """Normalize BLINK label like '(A)' -> 'A'. evaluation_local uses raw."""
    if not label:
        return label
    s = label.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s.strip().upper()


def _normalize_mmvp_label(label: str) -> str:
    if not label:
        return label
    s = label.strip()
    if s in ("(a)", "(b)", "(A)", "(B)"):
        return s.strip().upper()[1]
    return s.upper()


def _build_sample_iter(dataset, benchmark: str, image_dir: Optional[str]):
    """Yield (sample_id, image_paths, text, label, subset) per benchmark."""
    task_instruction = get_task_instruction(
        {"vstar": "vstar", "MMVP": "mmvp", "blink": "blink"}[benchmark]
    )

    for dat in dataset:
        if benchmark == "vstar":
            sample_id = str(dat["question_id"])
            img_path = os.path.join(image_dir, dat["image"])
            text = dat["text"] + task_instruction
            label = dat["label"]
            subset = dat.get("category")
            yield sample_id, img_path, text, label, subset
        elif benchmark == "MMVP":
            sample_id = str(dat["question_id"])
            img = dat["image"]
            if isinstance(img, Image.Image):
                img_path = img
            else:
                img_path = os.path.join(image_dir, img)
            label_raw = dat["label"]
            label = _normalize_mmvp_label(label_raw)
            text = dat["query"].replace("(a)", "A.").replace("(b)", "B.") + task_instruction
            yield sample_id, img_path, text, label, None
        elif benchmark == "blink":
            sample_id = str(dat["question_id"])
            img_path = dat["image"]  # already a list of PIL Images
            text = dat["query"] + task_instruction
            label = dat["label"]  # already uppercase letter
            subset = dat.get("category")
            yield sample_id, img_path, text, label, subset


def _run_batch(model, processor, batch_items, steps: int):
    """Run a batch through `model.generate(...)`. Returns the records dict.

    `batch_items` is a list of (img_path, text) tuples.
    """
    messages = [create_messages(img, txt) for img, txt in batch_items]
    text_formatted = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
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
    batch_size = len(batch_items)

    model._probe_records = {
        "iters": [],
        "prefill_hidden_last_layer": None,
        "prefill_input_ids": None,
        "prompt_len": None,
        "sequences": None,
    }

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy="steps",
            lvr_steps=[steps] * batch_size,
            plvr_target_only=False,
        )

    return model._probe_records


def run_extraction(args):
    print(f"[probe_extract] ckpt={args.ckpt}")
    print(f"[probe_extract] variant={args.variant} benchmark={args.benchmark} steps={args.steps}")
    model, processor = load_model_and_processor(args.ckpt)
    install_probe_instrumentation(model)
    tokenizer = processor.tokenizer
    special_ids = _build_special_ids(tokenizer)
    is_plvr = bool(getattr(model.config, "plvr_mode", False))
    print(f"[probe_extract] is_plvr={is_plvr}")

    # Use a dummy run_name (we don't need the eval output dir).
    dataset, image_dir, _out_dir, ds_name = BENCH_LOADERS[args.benchmark](
        getattr(model.config, "lvr_head", False), "probe_extract", "steps"
    )
    print(f"[probe_extract] {args.benchmark}: {len(dataset)} samples")

    if args.limit is not None and args.limit > 0:
        dataset = list(dataset)[: args.limit]
        print(f"[probe_extract] limited to {len(dataset)} samples")

    samples_out = []
    t0 = time.time()
    bench_for_iter = "MMVP" if args.benchmark.upper() == "MMVP" else args.benchmark
    sample_iter = list(_build_sample_iter(dataset, bench_for_iter, image_dir))

    bs = max(1, int(args.batch_size))

    # Batch the samples. Within a BLINK batch, samples may have multiple images
    # of varying counts; HF processor handles this uniformly via padding.
    pbar = tqdm(
        range(0, len(sample_iter), bs),
        desc=f"{args.variant}/{args.benchmark} (bs={bs})",
    )
    for start in pbar:
        chunk = sample_iter[start:start + bs]
        try:
            records = _run_batch(
                model,
                processor,
                [(img, text) for (_, img, text, _, _) in chunk],
                args.steps,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[warn] OOM at start={start} bs={bs}; falling back to per-sample for this chunk")
            # Process the chunk one sample at a time.
            for sample_tuple in chunk:
                sample_id, img_path, text, label, subset = sample_tuple
                try:
                    sub_records = _run_batch(model, processor, [(img_path, text)], args.steps)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[warn] OOM at sample {sample_id}; skipping")
                    continue
                sd = _extract_probes_from_records(sub_records, tokenizer, is_plvr, special_ids, 0)
                if sd is None:
                    print(f"[warn] sample {sample_id}: extraction failed (no LVR records)")
                    continue
                sd["sample_id"] = sample_id
                sd["subset"] = subset
                sd["label"] = _normalize_blink_label(label) if args.benchmark == "blink" else label
                samples_out.append(sd)
            continue

        for bi, sample_tuple in enumerate(chunk):
            sample_id, _img, _text, label, subset = sample_tuple
            sd = _extract_probes_from_records(records, tokenizer, is_plvr, special_ids, bi)
            if sd is None:
                print(f"[warn] sample {sample_id}: extraction failed (no LVR records)")
                continue
            sd["sample_id"] = sample_id
            sd["subset"] = subset
            sd["label"] = _normalize_blink_label(label) if args.benchmark == "blink" else label
            samples_out.append(sd)

    elapsed = time.time() - t0
    print(f"[probe_extract] done in {elapsed:.1f}s, {len(samples_out)} samples extracted")

    payload = {
        "variant": args.variant,
        "benchmark": args.benchmark,
        "checkpoint": args.ckpt,
        "seed": args.seed,
        "steps": args.steps,
        "is_plvr": is_plvr,
        "samples": samples_out,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = f"extract_{args.variant}_{args.benchmark}_seed{args.seed}.pt"
    out_path = os.path.join(args.out_dir, out_name)
    torch.save(payload, out_path)
    print(f"[probe_extract] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True,
                    help="Short variant tag, e.g. lvr_baseline, nlvr, plvr2, plvr3, dlvr_a.")
    ap.add_argument("--benchmark", required=True, choices=list(BENCH_LOADERS.keys()))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True,
                    help="Output directory; one .pt per (variant, benchmark, seed).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on number of samples (for smoke testing).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Generation batch size. H100 has plenty of memory; "
                         "8 is conservative, 16 may work for shorter prompts.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_extraction(args)


if __name__ == "__main__":
    main()
