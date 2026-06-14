"""Channel permutation helpers for channel-information diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def make_permutation(n_channels: int, seed: int) -> list[int]:
    """Create a reproducible channel permutation."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_channels).astype(int).tolist()


def _validate_permutation(perm: Sequence[int], n_channels: int) -> list[int]:
    perm = [int(idx) for idx in perm]
    if len(perm) != n_channels:
        raise ValueError(
            f"Permutation length {len(perm)} does not match {n_channels} channels."
        )
    if sorted(perm) != list(range(n_channels)):
        raise ValueError("Permutation must contain every channel index exactly once.")
    return perm


def permute_signal_channels(x: Any, perm: Sequence[int], channel_axis: int = 1) -> Any:
    """Permute the channel axis of a numpy array or torch Tensor."""
    n_channels = x.shape[channel_axis]
    perm = _validate_permutation(perm, n_channels)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for this helper.
        torch = None

    if torch is not None and isinstance(x, torch.Tensor):
        index = torch.as_tensor(perm, dtype=torch.long, device=x.device)
        return torch.index_select(x, dim=channel_axis, index=index)

    return np.take(x, indices=perm, axis=channel_axis)


def permute_channel_metadata(channel_meta: Any, perm: Sequence[int]) -> Any:
    """Apply the same channel permutation to common metadata containers."""
    if channel_meta is None:
        return None

    if isinstance(channel_meta, Mapping):
        permuted = {}
        for key, value in channel_meta.items():
            if _looks_channel_aligned(value, len(perm)):
                permuted[key] = permute_channel_metadata(value, perm)
            else:
                permuted[key] = value
        return permuted

    if isinstance(channel_meta, np.ndarray):
        _validate_permutation(perm, channel_meta.shape[0])
        return channel_meta[np.asarray(perm, dtype=int)]

    if isinstance(channel_meta, Sequence) and not isinstance(channel_meta, (str, bytes)):
        _validate_permutation(perm, len(channel_meta))
        return [channel_meta[idx] for idx in perm]

    raise TypeError(f"Unsupported channel metadata type: {type(channel_meta)!r}")


def apply_joint_permutation(
    x: Any,
    channel_meta: Any,
    perm: Sequence[int],
    channel_axis: int = 1,
) -> tuple[Any, Any]:
    """Permute EEG signal channels and channel metadata together.

    This helper preserves the physical signal-channel correspondence:
    x[:, i, ...] keeps the metadata that used to describe x[:, i, ...].
    It is not used by the CBraMod C3 runners because CBraMod does not consume
    explicit channel metadata in ``model.forward``.
    """
    x_perm = permute_signal_channels(x, perm=perm, channel_axis=channel_axis)
    meta_perm = permute_channel_metadata(channel_meta, perm=perm)
    return x_perm, meta_perm


def corrupt_channel_assignment_by_signal_permutation(
    x: Any,
    channel_meta: Any,
    perm: Sequence[int],
    channel_axis: int = 1,
) -> tuple[Any, Any]:
    """For implicit-order models: permute EEG signals, keep metadata fixed.

    CBraMod in this repository does not pass channel names/coordinates into
    ``model.forward``. The tensor channel index is therefore the model's implicit
    channel label. Permuting ``x`` while leaving metadata/model order unchanged
    assigns each signal to the wrong implicit channel label.
    """
    x_perm = permute_signal_channels(x, perm=perm, channel_axis=channel_axis)
    return x_perm, channel_meta


def corrupt_channel_assignment_by_metadata_permutation(
    x: Any,
    channel_meta: Any,
    perm: Sequence[int],
) -> tuple[Any, Any]:
    """For explicit-metadata models: permute metadata, keep EEG data fixed.

    Use this form for models that actually consume channel labels/coordinates.
    It is the literal "change channel labels while leaving EEG samples untouched"
    variant.
    """
    meta_perm = permute_channel_metadata(channel_meta, perm=perm)
    return x, meta_perm


def _looks_channel_aligned(value: Any, n_channels: int) -> bool:
    if isinstance(value, np.ndarray):
        return value.shape[:1] == (n_channels,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) == n_channels
    return False
