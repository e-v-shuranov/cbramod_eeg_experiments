"""Run the C2 joint signal-and-channel permutation diagnostic.

C2 is a harmless reordering check: the signal channels and their metadata are
permuted together. CBraMod does not consume explicit channel metadata in this
repository, so the actual model input change is the tensor channel order while
the metadata permutation is recorded in the output for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.channel_permutation import (
    apply_joint_permutation,
    make_permutation,
)


DATASET_NAMES = [
    "FACED",
    "SEED-V",
    "PhysioNet-MI",
    "SHU-MI",
    "ISRUC",
    "CHB-MIT",
    "BCIC2020-3",
    "Mumtaz2016",
    "SEED-VIG",
    "MentalArithmetic",
    "TUEV",
    "TUAB",
    "BCIC-IV-2a",
]

DATASETS: dict[str, tuple[object, object, str]] = {}


DEFAULT_CHANNEL_NAMES = {
    "TUEV": [
        "EEG FP1-REF",
        "EEG FP2-REF",
        "EEG F3-REF",
        "EEG F4-REF",
        "EEG C3-REF",
        "EEG C4-REF",
        "EEG P3-REF",
        "EEG P4-REF",
        "EEG O1-REF",
        "EEG O2-REF",
        "EEG F7-REF",
        "EEG F8-REF",
        "EEG T3-REF",
        "EEG T4-REF",
        "EEG T5-REF",
        "EEG T6-REF",
    ],
}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_channel_names(value: str | None, n_channels: int) -> list[str]:
    if value:
        if os.path.exists(value):
            with open(value, encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
        else:
            names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [f"ch_{idx}" for idx in range(n_channels)]

    if len(names) != n_channels:
        raise ValueError(
            f"Expected {n_channels} channel names, got {len(names)}."
        )
    return names


def build_params(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        cuda=args.cuda,
        epochs=0,
        batch_size=args.batch_size,
        lr=1e-4,
        weight_decay=5e-2,
        optimizer="AdamW",
        clip_value=-1,
        dropout=args.dropout,
        classifier=args.classifier,
        downstream_dataset=args.dataset,
        datasets_dir=args.datasets_dir,
        num_of_classes=args.num_of_classes,
        model_dir="",
        num_workers=args.num_workers,
        label_smoothing=0.0,
        multi_lr=False,
        frozen=False,
        use_pretrained_weights=args.use_pretrained_weights,
        foundation_dir=args.foundation_dir,
        use_cosine_warmup=False,
        use_scheduler=False,
        infer_only=True,
        model_for_test=args.checkpoint,
        n_chanels=args.n_channels,
        path_emb="",
        store_embedings=False,
        is_chanle_shafle=False,
        new_order=list(range(args.n_channels)),
        is_chanle_shafle_multitest=False,
        new_orders_list=[],
    )


def setup_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_project_modules() -> None:
    if DATASETS:
        return

    from datasets import (
        bciciv2a_dataset,
        chb_dataset,
        faced_dataset,
        isruc_dataset,
        mumtaz_dataset,
        physio_dataset,
        seedv_dataset,
        seedvig_dataset,
        shu_dataset,
        speech_dataset,
        stress_dataset,
        tuab_dataset,
        tuev_dataset,
    )
    from models import (
        model_for_bciciv2a,
        model_for_chb,
        model_for_faced,
        model_for_isruc,
        model_for_mumtaz,
        model_for_physio,
        model_for_seedv,
        model_for_seedvig,
        model_for_shu,
        model_for_speech,
        model_for_stress,
        model_for_tuab,
        model_for_tuev,
    )

    DATASETS.update(
        {
            "FACED": (faced_dataset, model_for_faced, "multiclass"),
            "SEED-V": (seedv_dataset, model_for_seedv, "multiclass"),
            "PhysioNet-MI": (physio_dataset, model_for_physio, "multiclass"),
            "SHU-MI": (shu_dataset, model_for_shu, "binary"),
            "ISRUC": (isruc_dataset, model_for_isruc, "multiclass"),
            "CHB-MIT": (chb_dataset, model_for_chb, "binary"),
            "BCIC2020-3": (speech_dataset, model_for_speech, "multiclass"),
            "Mumtaz2016": (mumtaz_dataset, model_for_mumtaz, "binary"),
            "SEED-VIG": (seedvig_dataset, model_for_seedvig, "regression"),
            "MentalArithmetic": (stress_dataset, model_for_stress, "binary"),
            "TUEV": (tuev_dataset, model_for_tuev, "multiclass"),
            "TUAB": (tuab_dataset, model_for_tuab, "binary"),
            "BCIC-IV-2a": (bciciv2a_dataset, model_for_bciciv2a, "multiclass"),
        }
    )


def load_model(args: argparse.Namespace, params: SimpleNamespace) -> Any:
    import torch

    _, model_module, _ = DATASETS[args.dataset]
    model = model_module.Model(params)
    map_location = torch.device(args.device)
    state_dict = _torch_load(args.checkpoint, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    return model


def _torch_load(path: str, map_location: torch.device):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_eval_loader(args: argparse.Namespace, params: SimpleNamespace):
    dataset_module, _, _ = DATASETS[args.dataset]
    loaders = dataset_module.LoadDataset(params).get_data_loader()
    return loaders[args.split]


def evaluate(
    model: Any,
    loader,
    task_type: str,
    device: str,
    perm: list[int] | None = None,
    channel_meta: dict | None = None,
) -> dict:
    import torch

    truths = []
    preds = []
    scores = []

    with torch.no_grad():
        for x, y, _file_names in loader:
            if perm is not None:
                x, _ = apply_joint_permutation(
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


def compute_metrics(
    task_type: str,
    truths: list,
    preds: list,
    scores: list,
) -> dict:
    from sklearn.metrics import (
        auc,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        r2_score,
        roc_auc_score,
    )

    truths_np = np.asarray(truths)
    preds_np = np.asarray(preds)

    if task_type == "multiclass":
        return {
            "score": float(balanced_accuracy_score(truths_np, preds_np)),
            "metric": "balanced_accuracy",
            "kappa": float(cohen_kappa_score(truths_np, preds_np)),
            "f1_weighted": float(f1_score(truths_np, preds_np, average="weighted")),
            "confusion_matrix": confusion_matrix(truths_np, preds_np).tolist(),
        }

    if task_type == "binary":
        scores_np = np.asarray(scores)
        precision, recall, _thresholds = precision_recall_curve(
            truths_np, scores_np, pos_label=1
        )
        return {
            "score": float(balanced_accuracy_score(truths_np, preds_np)),
            "metric": "balanced_accuracy",
            "roc_auc": float(roc_auc_score(truths_np, scores_np)),
            "pr_auc": float(auc(recall, precision)),
            "confusion_matrix": confusion_matrix(truths_np, preds_np).tolist(),
        }

    if task_type == "regression":
        return {
            "score": float(r2_score(truths_np, preds_np)),
            "metric": "r2",
        }

    raise ValueError(f"Unknown task type: {task_type}")


def write_rows(output: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "split",
        "score_original",
        "score_joint_permuted",
        "delta_joint_perm",
        "metric",
        "seed",
        "perm_seed",
        "permutation",
        "channel_names_original",
        "channel_names_joint_permuted",
        "checkpoint",
        "notes",
    ]
    mode = "a" if rows[0].get("append", False) else "w"
    write_header = mode == "w" or not os.path.exists(output)
    with open(output, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fieldnames}
            for row in rows
        )


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
    params = build_params(args)
    loader = load_eval_loader(args, params)
    model = load_model(args, params)
    task_type = DATASETS[args.dataset][2]

    channel_names = parse_channel_names(
        args.channel_names,
        args.n_channels,
    )
    if args.channel_names is None and args.dataset in DEFAULT_CHANNEL_NAMES:
        channel_names = DEFAULT_CHANNEL_NAMES[args.dataset]

    channel_meta = {"channel_names": channel_names}
    original = evaluate(
        model,
        loader,
        task_type=task_type,
        device=args.device,
        channel_meta=channel_meta,
    )

    rows = []
    for perm_seed in args.perm_seeds:
        perm = make_permutation(args.n_channels, perm_seed)
        _x_none, permuted_meta = apply_joint_permutation(
            np.zeros((1, args.n_channels, 1)),
            channel_meta=channel_meta,
            perm=perm,
            channel_axis=1,
        )
        permuted = evaluate(
            model,
            loader,
            task_type=task_type,
            device=args.device,
            perm=perm,
            channel_meta=channel_meta,
        )
        rows.append(
            {
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "score_original": original["score"],
                "score_joint_permuted": permuted["score"],
                "delta_joint_perm": original["score"] - permuted["score"],
                "metric": original["metric"],
                "seed": args.seed,
                "perm_seed": perm_seed,
                "permutation": json.dumps(perm),
                "channel_names_original": json.dumps(channel_names),
                "channel_names_joint_permuted": json.dumps(
                    permuted_meta["channel_names"]
                ),
                "checkpoint": args.checkpoint,
                "append": args.append,
                "notes": (
                    "CBraMod wrapper records jointly permuted metadata; this repo "
                    "does not pass channel metadata into the model forward call."
                ),
            }
        )

    write_rows(args.output, rows)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C2 joint signal-and-channel permutation diagnostic for CBraMod."
    )
    parser.add_argument("--model", default="CBraMod")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_NAMES))
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--foundation-dir", default="pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--use-pretrained-weights", action="store_true")
    parser.add_argument("--num-of-classes", type=int, required=True)
    parser.add_argument("--n-channels", type=int, default=16)
    parser.add_argument("--channel-names")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--perm-seeds", type=parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier", default="all_patch_reps")
    parser.add_argument(
        "--output",
        default="results/channel/c2_joint_permutation.csv",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to an existing CSV instead of overwriting it.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
