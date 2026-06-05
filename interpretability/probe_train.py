"""
Linear-probe training harness.

Reads the per-`(variant, benchmark, seed)` `.pt` caches written by
`probe_extract.py`, trains one stratified-5-fold linear classifier per
`(variant, benchmark, subset, position)` cell, and writes a single
`probe_results.json` summary.

Probe is a single linear layer (sklearn LogisticRegression with C=1.0,
max_iter=1000, l2 penalty). 

Run example:

    python interpretability/probe_train.py \
        --extract-dir ${WORKSPACE}/interpretability_results/probes_20260514 \
        --out ${WORKSPACE}/interpretability_results/probes_20260514/probe_results.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


POSITION_KEYS = [
    "hidden_a",
    "hidden_b",
    "hidden_c",
    "hidden_b_ctx",
    "hidden_c_ctx",
    "hidden_a1_question_mean",
    "hidden_a1_visual_mean",
]
POSITION_REPORT_NAMES = {
    "hidden_a": "a",
    "hidden_b": "b",
    "hidden_c": "c",
    "hidden_b_ctx": "b_ctx",
    "hidden_c_ctx": "c_ctx",
    "hidden_a1_question_mean": "a1_question_mean",
    "hidden_a1_visual_mean": "a1_visual_mean",
}


def _load_payloads(extract_dir: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(extract_dir, "extract_*.pt")))
    print(f"[probe_train] found {len(paths)} extract files in {extract_dir}")
    payloads = []
    for p in paths:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        payload["_path"] = p
        payloads.append(payload)
    return payloads


def _cells_from_payload(payload: Dict) -> Dict[Optional[str], List[Dict]]:
    """For a single payload, group samples by subset (None = whole benchmark)."""
    by_subset: Dict[Optional[str], List[Dict]] = defaultdict(list)
    # Always emit a None-keyed bucket containing all samples (for non-BLINK
    # benchmarks, and for the BLINK-aggregate row if we want it).
    benchmark = payload["benchmark"]
    for s in payload["samples"]:
        if benchmark == "blink":
            by_subset[s.get("subset")].append(s)
        else:
            by_subset[None].append(s)
    return by_subset


def _stack_position(samples: List[Dict], pos_key: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """Build X (n, H) and y (list of labels) for one position key.

    Samples lacking the position (e.g. hidden_b_ctx on single-stage) are dropped.
    Returns (None, []) if no samples have this position.
    """
    X_rows = []
    y_rows = []
    for s in samples:
        v = s.get(pos_key)
        if v is None:
            continue
        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        X_rows.append(v.astype(np.float32))
        y_rows.append(s["label"])
    if not X_rows:
        return None, []
    X = np.stack(X_rows, axis=0)
    return X, y_rows


def _train_cell(
    X: np.ndarray,
    y: List[str],
    k_fold: int,
    C: float,
    max_iter: int,
    seed: int,
) -> Optional[Dict]:
    """Run stratified k-fold CV. Returns metrics dict or None if infeasible."""
    if X is None or X.shape[0] < k_fold * 2:
        return None
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.shape[0] < 2:
        return None
    # If any class has fewer samples than k_fold, downgrade k.
    eff_k = int(min(k_fold, counts.min()))
    if eff_k < 2:
        return None

    # StandardScaler before logreg: hidden-state per-dim scales vary widely
    # (some dims have std ~3, others ~0.05), so LBFGS without normalization
    # spends most of its iterations adjusting per-feature scale. Standardizing
    # collapses convergence from O(1000) iter to O(50) iter for the same
    # accuracy. The probe remains a single linear layer.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=C,
            max_iter=max_iter,
            penalty="l2",
            solver="lbfgs",
            random_state=seed,
            n_jobs=1,
        )),
    ])
    skf = StratifiedKFold(n_splits=eff_k, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(clf, X, y_arr, cv=skf, scoring="accuracy", n_jobs=1)
        # Out-of-fold predicted probabilities → held-out cross-entropy.
        # CE is what gives the variational MI lower bound H(Y) - CE ≤ I(rep;Y);
        # accuracy is only related to it via Fano's inequality.
        proba = cross_val_predict(clf, X, y_arr, cv=skf, method="predict_proba", n_jobs=1)
        cls_ordering = sorted(np.unique(y_arr).tolist())
        ce = float(log_loss(y_arr, proba, labels=cls_ordering))
    except Exception as e:
        print(f"[probe_train] cross_val_score failed: {e}")
        return None
    chance = 1.0 / classes.shape[0]
    chance_ce = float(np.log(classes.shape[0]))  # entropy of uniform over the classes
    # Empirical label entropy H(Y) over this dataset — the correct upper bound
    # to subtract from CE for a formal variational MI lower bound. Uniform H
    # only applies when labels are exactly balanced.
    label_counts = np.bincount(np.searchsorted(np.sort(classes), y_arr))
    label_probs = label_counts / label_counts.sum()
    nonzero = label_probs[label_probs > 0]
    H_emp = float(-np.sum(nonzero * np.log(nonzero)))
    n_total = X.shape[0]
    n_test = int(round(n_total / eff_k))
    n_train = n_total - n_test
    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "cross_entropy": ce,
        "cross_entropy_chance": chance_ce,
        "label_entropy_empirical": H_emp,
        "mi_lower_bound_nats": max(0.0, H_emp - ce),
        "mi_lower_bound_uniform": max(0.0, chance_ce - ce),
        "chance": float(chance),
        "chance_corrected": float(np.mean(scores) - chance),
        "n_train": n_train,
        "n_test": n_test,
        "n_total": n_total,
        "k_fold_effective": eff_k,
        "n_classes": int(classes.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-fold", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payloads = _load_payloads(args.extract_dir)
    if not payloads:
        print("[probe_train] no payloads found; abort")
        sys.exit(1)

    results = []
    for payload in payloads:
        variant = payload["variant"]
        benchmark = payload["benchmark"]
        seed = payload["seed"]
        by_subset = _cells_from_payload(payload)

        print(f"\n[probe_train] === {variant} / {benchmark} (seed={seed}) ===")
        # For BLINK, also report a 'subset=ALL' aggregate row (raw accuracy
        # only — chance baseline is meaningless when answer spaces differ).
        if benchmark == "blink":
            all_samples = [s for ss in by_subset.values() for s in ss]
            by_subset["_ALL_"] = all_samples

        for subset, samples in by_subset.items():
            print(f"  subset={subset!r}: {len(samples)} samples")
            for pos_key in POSITION_KEYS:
                X, y = _stack_position(samples, pos_key)
                metrics = _train_cell(X, y, args.k_fold, args.C, args.max_iter, args.seed)
                if metrics is None:
                    continue
                row = {
                    "variant": variant,
                    "benchmark": benchmark,
                    "subset": subset,
                    "seed": int(seed),
                    "position": POSITION_REPORT_NAMES[pos_key],
                    "accuracy_mean": metrics["accuracy_mean"],
                    "accuracy_std": metrics["accuracy_std"],
                    "accuracy_seed_std": None,
                    "cross_entropy": metrics["cross_entropy"],
                    "cross_entropy_chance": metrics["cross_entropy_chance"],
                    "label_entropy_empirical": metrics["label_entropy_empirical"],
                    "mi_lower_bound_nats": metrics["mi_lower_bound_nats"],
                    "mi_lower_bound_uniform": metrics["mi_lower_bound_uniform"],
                    "chance": metrics["chance"],
                    "chance_corrected": metrics["chance_corrected"],
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                    "n_total": metrics["n_total"],
                    "k_fold_effective": metrics["k_fold_effective"],
                    "n_classes": metrics["n_classes"],
                }
                results.append(row)
                print(
                    f"    [{POSITION_REPORT_NAMES[pos_key]:<18}] "
                    f"acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f} "
                    f"CE={metrics['cross_entropy']:.3f} "
                    f"MI_LB={metrics['mi_lower_bound_nats']:.3f} nats "
                    f"chance={metrics['chance']:.3f} "
                    f"n={metrics['n_total']}"
                )

    output = {
        "config": {
            "k_fold": args.k_fold,
            "C": args.C,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[probe_train] wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
