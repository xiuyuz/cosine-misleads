# Cosine Misleads: Auxiliary Losses Reshape Vision Language Models, Not Their Latents

[![arXiv](https://img.shields.io/badge/arXiv-2606.05753-b31b1b.svg)](https://arxiv.org/abs/2606.05753)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Code release for the paper [*Cosine Misleads: Auxiliary Losses Reshape Vision Language Models, Not Their Latents*](https://arxiv.org/abs/2606.05753).

This repository contains the training, evaluation, and interpretability code used in the paper. 



## Quick Smoke Test

Before any GPU work, verify the release is self-consistent:
```bash
bash scripts/smoke_release.sh
```
This runs `py_compile` on every Python file, `bash -n` on every shell script, exercises the top-level imports, and reproduces the bootstrap-Pearson numbers from Appendix A. It does not require a GPU or any artifacts.

## Reproducing The Paper

### 1. Environment
```bash
pip install -r requirements.txt
```

Hardware used for the paper: NVIDIA H100 (80GB). One or two GPUs per training run.

### 2. Data
- **Training corpus**: Visual-CoT 438k. Download from <https://huggingface.co/datasets/deepcs233/Visual-CoT> and point `DATA_PATH` in each `scripts/finetune_*.sh` at the resulting `meta_data_lvr_sft_stage1.json`. We use the bounding-box-annotated SFT split.
- **V*Bench**: <https://github.com/penghao-wu/vstar>
- **MMVP**: <https://github.com/tsb0601/MMVP>
- **BLINK**: <https://huggingface.co/datasets/BLINK-Benchmark/BLINK>; we evaluate on the five validation subsets used by the codebase (Counting, IQ_Test, Jigsaw, Relative_Reflectance, Spatial_Relation), totaling 697 questions.

### 3. Train

Set `WORKSPACE` and `DATA_ROOT` in your shell. The scripts pick up Python and DeepSpeed from your PATH; override only if needed:
```bash
export WORKSPACE=/path/to/your/workspace
export DATA_ROOT=/path/to/your/data       # often the same as WORKSPACE
# Optional overrides (default uses `which python` / `which deepspeed`):
export PYTHON_BIN=$(which python)
export DEEPSPEED_BIN=$(which deepspeed)
```

Each script handles one variant. The variant-to-script mapping matches Table 1:

| Variant | Script |
|---|---|
| LVR (baseline)  | `scripts/finetune_lvr_stage1_3b.sh` |
| N-LVR           | `scripts/finetune_nlvr_stage1_3b.sh` |
| D-LVR           | `scripts/finetune_dlvr_a_stage1_3b.sh` (resumes from LVR step 1500) |
| P-LVR-2         | `INCLUDE_FREE_STAGE=False bash scripts/finetune_plvr_stage1_3b.sh` |
| P-LVR-3         | `INCLUDE_FREE_STAGE=True  bash scripts/finetune_plvr_stage1_3b.sh` (default) |

`--random_seed` in these scripts controls the data-packing RNG only (see Appendix C for the seed semantics caveat).

### 4. Evaluate
The eval harness reads paths from the `WORKSPACE` and `DATA_ROOT` env vars (see [evaluation_local.py](evaluation/evaluation_local.py)). It expects three benchmark trees under `${WORKSPACE}/eval_data/` for V*Bench, MMVP, and BLINK.
```bash
export WORKSPACE=/path/to/your/workspace
export DATA_ROOT=$WORKSPACE       # or wherever you mount HF cache
python evaluation/run_one_ckpt.py --ckpt $WORKSPACE/checkpoints/stage1/checkpoint-2500
# For P-LVR variants (per-stage step budget):
python evaluation/run_one_ckpt.py --ckpt <plvr-ckpt> --plvr
```

The harness writes per-prediction JSON files under `${WORKSPACE}/eval_results/{vstar,MMVP,blink}/decoding_by_steps/<slug>/`. The slug is derived from the checkpoint path. **Important**: V*Bench numbers in our paper use the `accuracy_reward` function defined in `evaluation/evaluation_local.py`, which extracts the substring between `<answer>` and `</answer>`, strips whitespace, and keeps the first character before comparing to the gold letter.

### 5. PRISM diagnostics

**Axis 1 — Linear probes** (probe_extract.py extracts hidden states; probe_train.py runs 5-fold logistic regression and reports accuracy + cross-entropy + MI lower bound):
```bash
bash scripts/run_probes.sh
# Or per-checkpoint:
python interpretability/probe_extract.py \
    --ckpt <checkpoint> --variant lvr_baseline --benchmark vstar --steps 8 --seed 42 \
    --out-dir $WORKSPACE/interpretability/probes_lvr --batch-size 1
python interpretability/probe_train.py \
    --extract-dir $WORKSPACE/interpretability/probes_lvr \
    --out $WORKSPACE/interpretability/probes_lvr/probe_results.json
```

`probe_train.py` reports both `mi_lower_bound_nats` (using the empirical V*Bench label entropy H(Y) = 1.267 nats; see Appendix A) and `mi_lower_bound_uniform` (using ln 4 = 1.386). The paper uses the empirical value.

**Axis 2 — Faithfulness corruption** (truncate, noise at three σ, random-donor swap):
```bash
bash scripts/run_faithfulness.sh
# Or per-checkpoint:
python interpretability/faithfulness_corrupt.py \
    --ckpt <checkpoint> --variant lvr_baseline --benchmark vstar --steps 8 --seed 42 \
    --out-dir $WORKSPACE/interpretability/faith_lvr --batch-size 1
```
All faithfulness runs use batch size 1 to match the main V*Bench eval (see Appendix B).

**Cosine to teacher target (Table 1 cosine column).** For each (checkpoint, sample) pair the script teacher-forces the visual targets at every LVR position, runs one forward pass with `output_hidden_states=True`, and computes `F.cosine_similarity(predicted_hidden, teacher_visual_embedding)` per LVR token. We report the mean over 200 sampled training instances. The cosine is raw (no centering or other transformation).
```bash
python interpretability/analysis_alignment.py \
    --ckpt <checkpoint> --config baseline \
    --data_path $WORKSPACE/lvr_data/meta_data_lvr_sft_stage1.json \
    --num_samples 200 --seed 42 \
    --output_dir $WORKSPACE/interpretability/cosine_lvr
# Use --config 2stage for P-LVR-2 and --config 3stage for P-LVR-3.
```

### 6. Reproduce Pearson statistics and figures
```bash
# Bootstrap CIs for cross-variant Pearson correlations (Appendix A)
python interpretability/bootstrap_pearson.py

# Figures
python paper/make_figures.py --faith-dir <faith_dir> --out-dir paper/figures
```

## Configuration Notes

- Absolute paths are parameterized as `${WORKSPACE}` / `${DATA_ROOT}` / `${HOME}`; set them to your own locations. The eval harness writes per-prediction JSON under slugs derived from the absolute checkpoint path (slash-replaced by underscores; see `evaluation/evaluation_local.py`).
- Wandb project names (`LVR-Qwen25-VL-3B-SFT-STAGE-1-450k`, etc.) describe the architecture; rename them to your own project as needed.
- Cloud-checkpointing code in `src/s3_checkpoints_lvr.py` and the OCI branches in `src/train/train_lvr.py` are dead code unless `--online_checkpoint True` is set; we ship them only for completeness.


## Mapping From Paper Sections To Code

| Paper element | Code |
|---|---|
| §3 IB view + Lagrangian | `src/train/monkey_patch_forward_lvr.py` (LVR/MSE loss assembly) |
| §5.1 variants (LVR, N-LVR, D-LVR, P-LVR-2, P-LVR-3) | `src/train/monkey_patch_forward_lvr.py` + the per-variant `scripts/finetune_*.sh` |
| §4 PRISM Axis 1 (linear probes at positions (a), (b)) | `interpretability/probe_extract.py`, `interpretability/probe_train.py` |
| §4 PRISM Axis 2 (faithfulness corruption) | `interpretability/faithfulness_corrupt.py` |
| Table 1 cosine column | `interpretability/analysis_alignment.py` |
| §5 V*Bench `accuracy_reward` parser | `evaluation/evaluation_local.py` |
| Appendix A Pearson bootstrap | `interpretability/bootstrap_pearson.py` |
| Figure 1 (teaser), faithfulness bars, gap-vs-corruption scatter | `paper/make_figures.py` |
| Appendix C training provenance | `scripts/finetune_*.sh` (one script per row of the provenance table) |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhang2026cosine,
      title={Cosine Misleads: Auxiliary Losses Reshape Vision Language Models, Not Their Latents}, 
      author={XiuYu Zhang and Junfeng Fang and Zhenkai Liang},
      year={2026},
      eprint={2606.05753},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.05753}, 
}
```

## Acknowledgements

The training of the LVR models and their variants is built on [VincentLeebang/lvr](https://github.com/VincentLeebang/lvr).
