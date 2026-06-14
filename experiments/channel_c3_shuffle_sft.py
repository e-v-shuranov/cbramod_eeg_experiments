"""Run C3 shuffle-then-fine-tune recovery evaluation.

C3 intentionally breaks channel assignment and then measures whether supervised
fine-tuning recovers performance. In this CBraMod repository channel metadata is
implicit in the tensor channel index, so corruption is implemented by permuting
the EEG signal channels while keeping the model's expected channel order fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from experiments.channel_c2_joint_perm import (
    DATASETS,
    DATASET_NAMES,
    build_params,
    load_eval_loader,
    load_model,
    load_project_modules,
    parse_channel_names,
    parse_int_list,
    setup_seed,
)
from experiments.channel_permutation import (
    corrupt_channel_assignment_by_signal_permutation,
    make_permutation,
)


def evaluate_with_optional_corruption(
    model,
    loader,
    task_type: str,
    device: str,
    channel_meta: dict,
    perm: list[int] | None,
) -> dict:
    import torch

    from experiments.channel_c2_joint_perm import compute_metrics

    truths = []
    preds = []
    scores = []

    with torch.no_grad():
        for x, y, _file_names in loader:
            if perm is not None:
                x, _ = corrupt_channel_assignment_by_signal_permutation(
                    x,
                    channel_meta=channel_meta,
                    perm=perm,
                    channel_axis=1,
                )
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)

            if task_type == "multiclass":
                pred_y = torch.max(pred, dim=-1)[1]
                truths.extend(y.cpu().squeeze().numpy().tolist())
                preds.extend(pred_y.cpu().squeeze().numpy().tolist())
            elif task_type == "binary":
                score_y = torch.sigmoid(pred)
                pred_y = torch.gt(score_y, 0.5).long()
                truths.extend(y.long().cpu().squeeze().numpy().tolist())
                preds.extend(pred_y.cpu().squeeze().numpy().tolist())
                scores.extend(score_y.cpu().numpy().reshape(-1).tolist())
            elif task_type == "regression":
                truths.extend(y.cpu().squeeze().numpy().tolist())
                preds.extend(pred.cpu().squeeze().numpy().tolist())
            else:
                raise ValueError(f"Unknown task type: {task_type}")

    return compute_metrics(task_type, truths, preds, scores)


def recovery_ratio(score_original: float, score_shuffled: float, score_sft: float):
    denom = score_original - score_shuffled
    if np.isclose(denom, 0.0):
        return None
    return (score_sft - score_shuffled) / denom


def write_rows(output: str, rows: list[dict], append: bool) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "split",
        "score_original",
        "score_shuffled",
        "score_shuffled_sft",
        "delta_shuffle",
        "r_shuffle",
        "metric",
        "seed",
        "perm_seed",
        "permutation",
        "baseline_checkpoint",
        "shuffled_sft_checkpoint",
        "finetune_epochs",
        "notes",
    ]
    mode = "a" if append else "w"
    write_header = mode == "w" or not os.path.exists(output)
    with open(output, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict]:
    import torch

    if args.dataset not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset {args.dataset!r}.")
    if args.model != "CBraMod":
        raise ValueError("This repository wrapper currently supports --model CBraMod.")

    load_project_modules()

    if args.device == "auto":
        args.device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.cuda)

    setup_seed(args.seed)

    args.use_pretrained_weights = False
    args.checkpoint = args.baseline_checkpoint
    baseline_params = build_params(args)
    baseline_params.model_for_test = args.baseline_checkpoint
    baseline_params.use_pretrained_weights = False
    baseline_model = load_model(
        argparse.Namespace(**{**vars(args), "checkpoint": args.baseline_checkpoint}),
        baseline_params,
    )

    args.checkpoint = args.shuffled_sft_checkpoint
    sft_params = build_params(args)
    sft_params.model_for_test = args.shuffled_sft_checkpoint
    sft_params.use_pretrained_weights = False
    sft_model = load_model(
        argparse.Namespace(**{**vars(args), "checkpoint": args.shuffled_sft_checkpoint}),
        sft_params,
    )

    loader = load_eval_loader(args, baseline_params)
    task_type = DATASETS[args.dataset][2]
    channel_names = parse_channel_names(args.channel_names, args.n_channels)
    channel_meta = {"channel_names": channel_names}

    rows = []
    for perm_seed in args.perm_seeds:
        perm = make_permutation(args.n_channels, perm_seed)
        original = evaluate_with_optional_corruption(
            baseline_model,
            loader,
            task_type=task_type,
            device=args.device,
            channel_meta=channel_meta,
            perm=None,
        )
        shuffled = evaluate_with_optional_corruption(
            baseline_model,
            loader,
            task_type=task_type,
            device=args.device,
            channel_meta=channel_meta,
            perm=perm,
        )
        shuffled_sft = evaluate_with_optional_corruption(
            sft_model,
            loader,
            task_type=task_type,
            device=args.device,
            channel_meta=channel_meta,
            perm=perm,
        )
        r_shuffle = recovery_ratio(
            original["score"],
            shuffled["score"],
            shuffled_sft["score"],
        )
        rows.append(
            {
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "score_original": original["score"],
                "score_shuffled": shuffled["score"],
                "score_shuffled_sft": shuffled_sft["score"],
                "delta_shuffle": original["score"] - shuffled["score"],
                "r_shuffle": "NA" if r_shuffle is None else r_shuffle,
                "metric": original["metric"],
                "seed": args.seed,
                "perm_seed": perm_seed,
                "permutation": json.dumps(perm),
                "baseline_checkpoint": args.baseline_checkpoint,
                "shuffled_sft_checkpoint": args.shuffled_sft_checkpoint,
                "finetune_epochs": args.finetune_epochs,
                "notes": (
                    "CBraMod has implicit channel labels; C3 corruption permutes "
                    "EEG signal channels while keeping model channel order fixed."
                ),
            }
        )

    write_rows(args.output, rows, append=args.append)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C3 shuffle-then-fine-tune recovery diagnostic for CBraMod."
    )
    parser.add_argument("--model", default="CBraMod")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_NAMES))
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--shuffled-sft-checkpoint", required=True)
    parser.add_argument("--foundation-dir", default="pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--num-of-classes", type=int, required=True)
    parser.add_argument("--n-channels", type=int, default=16)
    parser.add_argument("--channel-names")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--perm-seeds", type=parse_int_list, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier", default="all_patch_reps")
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/channel/c3_shuffle_sft_recovery.csv",
    )
    parser.add_argument("--append", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
