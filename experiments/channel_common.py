"""Shared utilities for CBraMod channel diagnostics."""

from __future__ import annotations

import argparse
import random
from types import SimpleNamespace
from typing import Any

import numpy as np


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


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_channel_names(value: str | None, n_channels: int) -> list[str]:
    if value:
        import os

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
        use_pretrained_weights=getattr(args, "use_pretrained_weights", False),
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


def _torch_load(path: str, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_eval_loader(args: argparse.Namespace, params: SimpleNamespace):
    dataset_module, _, _ = DATASETS[args.dataset]
    loaders = dataset_module.LoadDataset(params).get_data_loader()
    return loaders[args.split]


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
