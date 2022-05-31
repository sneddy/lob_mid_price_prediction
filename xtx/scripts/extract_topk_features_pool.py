import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats
from contexttimer import Timer
from pandarallel import pandarallel

import xtx.utils.dev_utils as dev_utils
from xtx.features.flatten_tools import ask_flatten, bid_flatten

pandarallel.initialize(progress_bar=False)
logger = dev_utils.init_logger("logging/flatten_building.log")


def fix_ewm(flatten_df, features, n, prefix):
    weights = np.exp(np.linspace(0.0, -1, n))
    weights /= weights.sum()
    features[f"{prefix}_ewm_{n}"] = flatten_df.parallel_apply(lambda x: np.dot(x, weights[: len(x)]))
    ewm_rest_mapping = {idx: weights[idx:].sum() for idx in range(n + 1)}
    last_elem_series = flatten_df.parallel_apply(lambda x: x[-1])
    features[f"{prefix}_ewm_{n}"] += features[f"{prefix}_len_{n}"].map(ewm_rest_mapping) * last_elem_series


def add_flatten_features(flatten_df: pd.DataFrame, features: pd.DataFrame, n: int, prefix: str):
    weights = np.exp(np.linspace(0.0, -1, n))
    weights /= weights.sum()

    features[f"{prefix}_len_{n}"] = flatten_df.parallel_apply(len)
    features[f"{prefix}_ewm_{n}"] = flatten_df.parallel_apply(lambda x: np.dot(x, weights[: len(x)]))
    ewm_rest_mapping = {idx: weights[idx:].sum() for idx in range(n + 1)}
    last_elem_series = flatten_df.parallel_apply(lambda x: x[-1])
    features[f"{prefix}_ewm_{n}"] += features[f"{prefix}_len_{n}"].map(ewm_rest_mapping) * last_elem_series

    (n - features[f"{prefix}_len_{n}"])
    features[f"{prefix}_mean_{n}"] = flatten_df.parallel_apply(np.mean)
    features[f"{prefix}_std_{n}"] = flatten_df.parallel_apply(np.std)
    features[f"{prefix}_iqr_{n}"] = flatten_df.parallel_apply(scipy.stats.iqr)
    features[f"{prefix}_skew_{n}"] = flatten_df.parallel_apply(scipy.stats.skew)
    features[f"{prefix}_kurtosis_{n}"] = flatten_df.parallel_apply(scipy.stats.kurtosis)


def extract_flatten_features(data: pd.DataFrame, n: int) -> pd.DataFrame():
    """
    Args:
        data (pd.DataFrame): _description_
        n (int): number of operation from every moment. Other ignore or fillna

    Returns:
        pd.DataFrame(): corresponding features
    """
    with Timer() as ask_flatten_time:
        ask_flatten_df = data.parallel_apply(lambda x: ask_flatten(x, n=n), axis=1)
    logger.info(f"Ask {n} flatten time: {ask_flatten_time.elapsed:.4f} sec")
    logger.info(ask_flatten_df.sample(10))

    with Timer() as bid_flatten_time:
        bid_flatten_df = data.parallel_apply(lambda x: bid_flatten(x, n=n), axis=1)
    logger.info(f"Bid {n} flatten time: {bid_flatten_time.elapsed:.4f} sec")
    logger.info(bid_flatten_df.sample(10))

    features = pd.DataFrame()
    with Timer() as ask_features_time:
        add_flatten_features(ask_flatten_df, features, n, prefix="ask_flatten")
    logger.info(f"Base {n} ask features time: {ask_features_time.elapsed:.4f} sec")
    logger.info(features.sample(10))

    with Timer() as bid_features_time:
        add_flatten_features(bid_flatten_df, features, n, prefix="bid_flatten")
    logger.info(f"Base {n} bid features time: {bid_features_time.elapsed:.4f} sec")
    logger.info(features.sample(10))

    with Timer() as interaction_time:
        features[f"flatten_spread_{n}_mean"] = (
            features[f"ask_flatten_mean_{n}"] / features[f"bid_flatten_mean_{n}"] - 1
        )
        features[f"flatten_spread_{n}_ewm"] = features[f"ask_flatten_ewm_{n}"] / features[f"bid_flatten_ewm_{n}"] - 1
        features[f"wap_flatten_{n}"] = (
            (features[f"ask_flatten_mean_{n}"] * features[f"bid_flatten_len_{n}"])
            + (features[f"bid_flatten_mean_{n}"] * features[f"ask_flatten_len_{n}"])
        ) / (features[f"ask_flatten_len_{n}"] + features[f"bid_flatten_len_{n}"])

        features[f"ewm_wap_flatten_{n}"] = (
            (features[f"ask_flatten_ewm_{n}"] * features[f"bid_flatten_len_{n}"])
            + (features[f"bid_flatten_ewm_{n}"] * features[f"ask_flatten_len_{n}"])
        ) / (features[f"ask_flatten_len_{n}"] + features[f"bid_flatten_len_{n}"])
    logger.info(f"Feature interaction time: {interaction_time.elapsed:.4f} sec")
    return ask_flatten_df, bid_flatten_df, features


def run(data_path, features_dir):
    data = pd.read_csv(data_path)
    os.makedirs(features_dir, exist_ok=True)
    for n_per_row in (5, 25, 50, 100):
        ask_flatten_df, bid_flatten_df, features = extract_flatten_features(data, n=n_per_row)
        features.to_pickle(f"{features_dir}/features_{n_per_row}.pkl")
        ask_flatten_df.to_pickle(f"{features_dir}/ask_flatten_{n_per_row}.pkl")
        bid_flatten_df.to_pickle(f"{features_dir}/bid_flatten_{n_per_row}.pkl")
        features.to_csv(f"{features_dir}/features_{n_per_row}.csv")


def argparser():
    parser = argparse.ArgumentParser(description="Xtx pipeline")
    parser.add_argument("train_cfg", type=str, help="train config path")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    experiment_config_path = args.train_cfg.strip("/")
    experiment = dev_utils.load_yaml(experiment_config_path)
    run(data_path=experiment["train_data_path"], features_dir=experiment["train_topk_features"])
    run(data_path=experiment["test_data_path"], features_dir=experiment["test_topk_features"])
