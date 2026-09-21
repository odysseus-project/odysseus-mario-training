from verl.utils.dataset import RLHFDataset

import datasets


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset('parquet', data_files=parquet_file)["train"]
            dataframe = dataframe.map(self.map_fn, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict):
        return row


def test_game_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # This is only called at the end of rollout
    # **kwargs allows accepting additional parameters like reward_router_address, reward_model_tokenizer
    # that may be passed by the reward loop but are not used by this function
    if data_source == "super_mario_land":
        multiplier = 1.0
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    game_score = extra_info["turn_scores"] * multiplier
    result = {"score": game_score, "pred": solution_str}
    # print(f"Return game scores: {game_score}")
    if "level_progress" in extra_info:
        level_progress = extra_info["level_progress"]
        result["level_progress"] = level_progress
    if "game_score" in extra_info:
        game_score = extra_info["game_score"]
        result["game_score"] = game_score
    return result
