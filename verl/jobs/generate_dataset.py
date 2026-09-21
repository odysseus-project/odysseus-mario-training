# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate Super Mario Land rollout inputs in Parquet format.
"""

import argparse
from pathlib import Path

GAMES = ("super_mario_land",)


def make_sample(game, prompt, index):
    return {
        "data_source": game,
        "prompt": [{"role": "system", "content": prompt}],
        "ability": "game",
        "reward_model": {"style": "rule", "ground_truth": None},
        "agent_name": "game_agent",
        "sample_index": index,
    }


def generate_dataset(output_dir, train_size=1024, test_size=128, game="super_mario_land", overwrite=False):
    if game not in GAMES:
        raise ValueError(f"Unknown game: {game}")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Both train_size and test_size must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    targets = [output_dir / "train.parquet", output_dir / "test.parquet"]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"Dataset already exists in {output_dir}; pass --overwrite to replace it")

    import datasets
    from verl.utils.game_envs.super_mario_land.env_prompts import SYSTEM_PROMPT

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, size in zip(targets, (train_size, test_size), strict=True):
        data = datasets.Dataset.from_list([make_sample(game, SYSTEM_PROMPT, i) for i in range(size)])
        data.to_parquet(str(path))
        print(f"Wrote {size} rows to {path}")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=GAMES, default="super_mario_land")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output_dir", type=Path, help="Write train.parquet and test.parquet directly here")
    output.add_argument("--local_dir", type=Path, help="Parent directory; writes to <game>_markov beneath it")
    parser.add_argument("--train_size", "-n", type=int, default=1024)
    parser.add_argument("--test_size", "-m", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    default_root = Path(__file__).resolve().parent / "dataset"
    output_dir = args.output_dir or (args.local_dir or default_root) / f"{args.game}_markov"
    if args.train_size <= 0 or args.test_size <= 0:
        parser.error("Both --train_size and --test_size must be positive")
    generate_dataset(output_dir, args.train_size, args.test_size, args.game, args.overwrite)


if __name__ == "__main__":
    main()
