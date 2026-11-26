import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


CHINESE_DATASETS = ("sims", "simsv2", "unic")
ENGLISH_DATASETS = ("mosi", "mosei")
DATASETS = (*CHINESE_DATASETS, *ENGLISH_DATASETS)
BACKBONES = {
    "transformer": "Joint Transformer",
    "mult": "MulT",
    "perceiver_io": "Perceiver IO",
    "pitf": "PITF",
}
FACTORIAL_ARMS = tuple(f"{index:03b}" for index in range(8))
SEMANTIC_METHODS = ("Backbone", "Polarity Only", "Intensity Only", "PITF")
FULL_ARMS = {"111", "pitf"}
SEMANTIC_ARMS = {
    "000": "Backbone",
    "polarity_only": "Polarity Only",
    "pitf_polarity_only": "Polarity Only",
    "intensity_only": "Intensity Only",
    "pitf_intensity_only": "Intensity Only",
    "111": "PITF",
    "pitf": "PITF",
}
METRIC_COLUMNS = {
    "I": ("acc2", "f1_weighted", "mae", "corr"),
    "II": ("acc2", "acc2_nonzero", "f1_weighted", "f1_nonzero", "acc7", "mae", "corr"),
    "III": ("acc2", "f1_weighted", "acc3", "acc5", "mae", "corr"),
    "IV": ("acc2", "f1_weighted", "acc3", "acc5", "mae", "corr"),
    "VI": ("acc2", "f1_weighted", "acc3", "acc5", "mae", "corr"),
}
CONFLICT_COLUMNS = (
    "dataset", "subset", "n", "seed_pairs", "runs",
    "ldw_acc2", "pi_acc2", "ldw_f1_weighted", "pi_f1_weighted",
    "ldw_mae", "pi_mae", "delta_mae",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dataset_name(path, summary):
    dataset = summary.get("dataset")
    if dataset is None and path.parent.name in DATASETS:
        return path.parent.name
    return dataset


def model_rows(summary):
    if "rows" in summary:
        yield from summary["rows"]
    else:
        for group in summary["groups"].values():
            yield from group["rows"]


def row_metrics(row):
    metrics = row.get("metrics", row)
    return metrics.get("M", metrics)


def display_metric(name, value):
    decimals = 6
    scale = 100 if name.startswith(("acc", "f1")) else 1
    return round(value * scale, decimals)


def select_method(table, dataset, row):
    arm = row["arm"]
    if table == "I" and dataset in ("sims", "simsv2") and arm in FULL_ARMS:
        return "", "PITF", arm
    if table == "II" and dataset in ENGLISH_DATASETS and arm in FULL_ARMS:
        return "", "PITF", arm
    if table == "III" and dataset in CHINESE_DATASETS and arm in FACTORIAL_ARMS:
        return "", arm, arm
    if table == "IV" and dataset in CHINESE_DATASETS and arm in SEMANTIC_ARMS:
        return "", SEMANTIC_ARMS[arm], arm
    if table == "VI" and dataset in CHINESE_DATASETS:
        backbone = row.get("backbone")
        if backbone in BACKBONES and arm in ("A", "C"):
            return backbone, {"A": "Baseline", "C": "+ PI-LDW"}[arm], arm
        if arm in ("011", "111"):
            return "pitf", {"011": "Baseline", "111": "+ PI-LDW"}[arm], arm
    return None


def aggregate_models(table, inputs, requested_seed):
    groups = defaultdict(dict)
    for path, summary in inputs:
        fallback_dataset = dataset_name(path, summary)
        for row in model_rows(summary):
            dataset = row.get("dataset", fallback_dataset)
            if dataset not in DATASETS:
                raise ValueError(f"Dataset missing or unknown in {path}.")
            seed = int(row["seed"])
            if requested_seed is not None and requested_seed != seed:
                continue
            selection = select_method(table, dataset, row)
            if selection is None:
                continue
            backbone, method, arm = selection
            group = groups[dataset, backbone, method]
            value = arm, row_metrics(row)
            if seed in group and group[seed] != value:
                raise ValueError(f"Multiple results for {dataset} {backbone} {method}, seed {seed}.")
            group[seed] = value

    method_order = (*FACTORIAL_ARMS, *SEMANTIC_METHODS, "Baseline", "+ PI-LDW")
    backbone_order = ("", *BACKBONES)
    order = lambda key: (
        DATASETS.index(key[0]), backbone_order.index(key[1]), method_order.index(key[2])
    )
    columns = METRIC_COLUMNS[table]
    for dataset, backbone, method in sorted(groups, key=order):
        group = groups[dataset, backbone, method]
        values = [item[1] for item in group.values()]
        result = {
            "dataset": dataset,
            "method": method,
            "arm": ",".join(sorted({item[0] for item in group.values()})),
            "seeds": ",".join(str(seed) for seed in sorted(group)),
            "runs": len(group),
            **{name: display_metric(name, mean(value[name] for value in values)) for name in columns},
        }
        if table == "VI":
            result["backbone"] = BACKBONES[backbone]
        if table == "III":
            result.update(dict(zip("ABC", method)))
        yield result


def parse_pair(value):
    try:
        dataset, ldw_seed, pi_seed = value.split(":")
        pair = int(ldw_seed), int(pi_seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use DATASET:LDW_SEED:PI_SEED.") from error
    if dataset not in CHINESE_DATASETS:
        raise argparse.ArgumentTypeError("Table V dataset must be sims, simsv2 or unic.")
    return dataset, pair


def conflict_reports(inputs, pairs, requested_seed):
    from .conflict import SUBSETS, compare_predictions

    reports = defaultdict(dict)
    for path, summary in inputs:
        dataset = dataset_name(path, summary)
        if dataset not in CHINESE_DATASETS:
            raise ValueError(f"Table V requires sims, simsv2 or unic input: {path}.")
        if all(subset in summary for subset in SUBSETS):
            pair = summary.get("ldw_seed"), summary.get("pi_seed")
            if dataset in pairs and pair != pairs[dataset]:
                continue
            if requested_seed is not None and pair != (requested_seed, requested_seed):
                continue
            if pair in reports[dataset]:
                raise ValueError(f"Multiple conflict results for {dataset}, seed pair {pair}.")
            reports[dataset][pair] = summary
            continue

        groups = summary["groups"]
        ldw_arm = "pitf_absolute" if "pitf_absolute" in groups else "absolute"
        pi_arm = next((arm for arm in ("111", "pitf", "polarity_intensity") if arm in groups), None)
        if ldw_arm not in groups or pi_arm is None:
            raise ValueError(f"Table V requires LDW and PI-LDW results in {path}.")
        ldw_seeds = {int(row["seed"]) for row in groups[ldw_arm]["rows"]}
        pi_seeds = {int(row["seed"]) for row in groups[pi_arm]["rows"]}
        if dataset in pairs:
            candidates = [pairs[dataset]]
        elif requested_seed is not None:
            candidates = [(requested_seed, requested_seed)]
        else:
            candidates = [(seed, seed) for seed in sorted(ldw_seeds & pi_seeds)]
        for ldw_seed, pi_seed in candidates:
            if ldw_seed not in ldw_seeds or pi_seed not in pi_seeds:
                raise ValueError(f"Requested seed pair is absent in {path}: {ldw_seed}:{pi_seed}.")
            if (ldw_seed, pi_seed) in reports[dataset]:
                raise ValueError(f"Multiple conflict results for {dataset}, seed pair {ldw_seed}:{pi_seed}.")
            ldw = path.parent / "arms" / ldw_arm / f"seed_{ldw_seed}" / "test_predictions.npz"
            pi = path.parent / "arms" / pi_arm / f"seed_{pi_seed}" / "test_predictions.npz"
            reports[dataset][ldw_seed, pi_seed] = compare_predictions(ldw, pi)
    return reports


def aggregate_conflict(inputs, pairs, requested_seed):
    from .conflict import SUBSETS

    reports = conflict_reports(inputs, pairs, requested_seed)
    for dataset in CHINESE_DATASETS:
        group = reports.get(dataset)
        if not group:
            continue
        pair_labels = [f"{ldw}:{pi}" for ldw, pi in group if ldw is not None and pi is not None]
        for subset in SUBSETS:
            values = [report[subset] for report in group.values()]
            counts = {value["n"] for value in values}
            if len(counts) != 1:
                raise ValueError(f"{dataset} {subset} report sample counts differ.")
            count = counts.pop()
            row = {
                "dataset": dataset, "subset": subset, "n": count,
                "seed_pairs": ",".join(pair_labels), "runs": len(group),
            }
            if count:
                for name, key in (("ldw", "LDW"), ("pi", "PI")):
                    for metric in ("acc2", "f1_weighted", "mae"):
                        row[f"{name}_{metric}"] = display_metric(
                            metric, mean(value[key][metric] for value in values)
                        )
                row["delta_mae"] = display_metric("mae", mean(value["delta_MAE"] for value in values))
            yield row


def main():
    parser = argparse.ArgumentParser(description='Export PITF experiment rows for paper Tables I–VI.')
    parser.add_argument("--table", required=True, choices=("I", "II", "III", "IV", "V", "VI"))
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seed", type=int, help="Use one seed instead of averaging.")
    selection.add_argument(
        "--pairs", nargs="+", type=parse_pair, metavar="DATASET:LDW_SEED:PI_SEED",
        help="Use explicit model seed pairs for Table V.",
    )
    args = parser.parse_args()
    inputs = [(path, load_json(path)) for path in args.input]
    if args.table == "V":
        pairs = dict(args.pairs or [])
        if args.pairs and len(pairs) != len(args.pairs):
            parser.error("Specify at most one seed pair per dataset.")
        rows = list(aggregate_conflict(inputs, pairs, args.seed))
        if pairs.keys() - {row["dataset"] for row in rows}:
            parser.error("An explicit seed pair has no matching input.")
        columns = CONFLICT_COLUMNS
    else:
        if args.pairs:
            parser.error("--pairs applies to Table V.")
        rows = list(aggregate_models(args.table, inputs, args.seed))
        prefix = {
            "III": ("dataset", "A", "B", "C"),
            "VI": ("backbone", "dataset"),
        }.get(args.table, ("dataset",))
        columns = (*prefix, "method", "arm", "seeds", "runs", *METRIC_COLUMNS[args.table])
    if not rows:
        parser.error("No matching experiment rows in the supplied inputs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
