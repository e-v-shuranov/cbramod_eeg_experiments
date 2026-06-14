"""Export deterministic channel shuffle plans as CSV.

This is a lightweight helper for manuscript reporting: it records each
permutation both as channel indices and as channel names.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from experiments.channel_c2_joint_perm import parse_channel_names, parse_int_list
from experiments.channel_permutation import make_permutation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export channel shuffle plans.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-channels", type=int, required=True)
    parser.add_argument("--channel-names", required=True)
    parser.add_argument("--perm-seeds", type=parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channel_names = parse_channel_names(args.channel_names, args.n_channels)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "perm_seed",
            "permutation_indices",
            "channel_names_original",
            "channel_names_after_shuffle",
            "index_to_name_mapping",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for perm_seed in args.perm_seeds:
            perm = make_permutation(args.n_channels, perm_seed)
            names_after_shuffle = [channel_names[idx] for idx in perm]
            index_to_name_mapping = [
                {
                    "model_position": position,
                    "source_channel_index": source_idx,
                    "source_channel_name": channel_names[source_idx],
                }
                for position, source_idx in enumerate(perm)
            ]
            writer.writerow(
                {
                    "dataset": args.dataset,
                    "perm_seed": perm_seed,
                    "permutation_indices": json.dumps(perm),
                    "channel_names_original": json.dumps(channel_names),
                    "channel_names_after_shuffle": json.dumps(names_after_shuffle),
                    "index_to_name_mapping": json.dumps(index_to_name_mapping),
                }
            )

    print(f"Wrote shuffle plan to {args.output}")


if __name__ == "__main__":
    main()
