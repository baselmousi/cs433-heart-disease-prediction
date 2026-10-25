"""Compare cross-validation runs fold by fold, from their saved out-of-fold scores.

The first name is the reference. Every other run is compared with it on the same
folds (a paired comparison), which removes the fold-to-fold noise the two runs share.
With several seeds, the change of one run's mean F1 between seeds shows how much the
fold split alone moves a reported score. Run from the repository root after run_cv.py:

    python compare_cv.py E1-logistic E3-indicators E9-leaky
    python compare_cv.py E1-logistic E9-leaky --seeds 42 43
"""

import argparse
from pathlib import Path

import numpy as np

from run_cv import run_name
from src.cross_validation import (
    paired_comparison,
    per_fold_f1,
    stratified_kfold_indices,
)

ROOT = Path(__file__).resolve().parent


def load_run(oof_dir, name, seed):
    """Out-of-fold probabilities and labels of one run; files saved before the
    --seed option have no seed and used 42."""
    path = Path(oof_dir) / f"{run_name(name, seed)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run run_cv.py {name} --seed {seed}")
    data = np.load(path)
    saved_seed = int(data["seed"]) if "seed" in data.files else 42
    if saved_seed != seed:
        raise ValueError(f"{path} was run with seed {saved_seed}, not {seed}.")
    return data["probabilities"], data["labels"]


def compare(oof_dir, names, seed):
    """Print each run's F1 and its paired difference from the first run; return the
    mean F1 of every run."""
    runs = {name: load_run(oof_dir, name, seed) for name in names}
    labels = runs[names[0]][1]
    for name, (_, other_labels) in runs.items():
        if not np.array_equal(other_labels, labels):
            raise ValueError(f"{name} was scored on different labels.")
    folds = stratified_kfold_indices((labels == 1).astype(float), seed=seed)
    base = runs[names[0]][0]

    print(f"\nSeed {seed}: differences are F1(run) - F1({names[0]}) per fold")
    print(
        f"{'run':<20} {'F1':>7} {'cutoff':>6}   {'per-fold difference':<44} "
        f"{'mean':>8} {'t':>6}  better  verdict"
    )
    means = {}
    for name in names:
        threshold, scores = per_fold_f1(labels, runs[name][0], folds)
        means[name] = float(np.mean(scores))
        start = f"{name:<20} {means[name]:7.4f} {threshold:6.2f}"
        if name == names[0]:
            print(start)
            continue
        result = paired_comparison(labels, base, runs[name][0], folds)
        differences = " ".join(f"{d:+.4f}" for d in result["differences"])
        print(
            f"{start}   {differences:<44} {result['mean']:+8.4f} "
            f"{result['t']:6.1f}  {result['folds_better']}/{len(scores)}     "
            f"{result['verdict']}"
        )
    return means


def main():
    """Command-line entry point; reads results/oof by default."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs", nargs="+", help="experiment names; first is the reference"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "results" / "oof")
    args = parser.parse_args()

    means = {seed: compare(args.oof_dir, args.runs, seed) for seed in args.seeds}
    if len(args.seeds) > 1:
        print("\nMean F1 by seed (the spread is the noise from the fold split)")
        print(f"{'run':<20} " + " ".join(f"{s:>8}" for s in args.seeds) + "   spread")
        for name in args.runs:
            values = [means[s][name] for s in args.seeds]
            print(
                f"{name:<20} "
                + " ".join(f"{v:8.4f}" for v in values)
                + f"   {max(values) - min(values):.4f}"
            )


if __name__ == "__main__":
    main()
