import argparse
import json
from pathlib import Path

import numpy as np

from reproductions.confede.metrics import sentiment_metrics


DATASETS = ("sims", "simsv2", "unic")
SUBSETS = ("Non-conflict", "Conflict", "Overall")


def compare_predictions(ldw_path, pi_path):
    with np.load(ldw_path) as ldw, np.load(pi_path) as pi:
        ldw_ids, pi_ids = ldw["ids"], pi["ids"]
        positions = {sample_id: index for index, sample_id in enumerate(pi_ids)}
        if len(set(ldw_ids)) != len(ldw_ids) or len(positions) != len(pi_ids):
            raise ValueError("Prediction files contain duplicate sample IDs.")
        if set(ldw_ids) != set(positions):
            raise ValueError("Prediction files contain different sample IDs.")
        order = [positions[sample_id] for sample_id in ldw_ids]
        labels = {}
        for modality in "MTAV":
            field = f"truth_{modality}"
            labels[modality] = ldw[field].reshape(-1)
            np.testing.assert_array_equal(labels[modality], pi[field].reshape(-1)[order])
        if any(not np.all(np.abs(values) <= 1) for values in labels.values()):
            raise ValueError("Conflict analysis requires M/T/A/V labels in [-1, 1].")
        truth = labels["M"]
        conflict = np.any(np.column_stack([labels[mode] * truth < 0 for mode in "TAV"]), axis=1)
        ldw_pred = ldw["M"].reshape(-1)
        pi_pred = pi["M"].reshape(-1)[order]
        truth64 = truth.astype(np.float64)
        ldw_error = np.abs(truth64 - np.clip(ldw_pred.astype(np.float64), -1, 1))
        pi_error = np.abs(truth64 - np.clip(pi_pred.astype(np.float64), -1, 1))
        gain = ldw_error - pi_error

    report = {}
    for name, mask in zip(SUBSETS, (~conflict, conflict, np.ones(len(truth), dtype=bool))):
        size = int(mask.sum())
        report[name] = {
            "n": size,
            "LDW": sentiment_metrics(truth[mask], ldw_pred[mask]) if size else {},
            "PI": sentiment_metrics(truth[mask], pi_pred[mask]) if size else {},
            "delta_MAE": float(gain[mask].mean()) if size else None,
        }
    report["G"] = (
        report["Conflict"]["delta_MAE"] - report["Non-conflict"]["delta_MAE"]
        if conflict.any() and (~conflict).any()
        else None
    )
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare LDW and PI-LDW on the same test samples.')
    parser.add_argument("--ldw", required=True, type=Path)
    parser.add_argument("--pi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--ldw-seed", type=int)
    parser.add_argument("--pi-seed", type=int)
    args = parser.parse_args()
    report = compare_predictions(args.ldw, args.pi)
    report.update(dataset=args.dataset, ldw_seed=args.ldw_seed, pi_seed=args.pi_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
