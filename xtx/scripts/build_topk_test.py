import os

import numpy as np
import pandas as pd
import scipy.stats
from contexttimer import Timer
from pandarallel import pandarallel

from xtx.features.flatten_tools import ask_flatten, bid_flatten
from xtx.utils.dev_utils import init_logger

pandarallel.initialize(progress_bar=False)
logger = init_logger("logging/flatten_building.log")


def read_data(n_global: int = None):
    data = pd.read_pickle("data/data.csv")
    if n_global is None:
        return data
    return data.head(n_global)


def add_flatten_features(flatten_df: pd.DataFrame, features: pd.DataFrame, n: int, prefix: str):
    features[f"{prefix}_len_{n}"] = flatten_df.parallel_apply(len)
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
    logger.info(f"Ask flatten time: {ask_flatten_time.elapsed:.4f} sec")
    logger.info(ask_flatten_df.sample(10))

    with Timer() as bid_flatten_time:
        bid_flatten_df = data.parallel_apply(lambda x: bid_flatten(x, n=n), axis=1)
    logger.info(f"Bid flatten time: {bid_flatten_time.elapsed:.4f} sec")
    logger.info(bid_flatten_df.sample(10))

    features = pd.DataFrame()
    with Timer() as ask_features_time:
        add_flatten_features(ask_flatten_df, features, n, prefix="ask_flatten")
    logger.info(f"Base ask features time: {ask_features_time.elapsed:.4f} sec")
    logger.info(features.sample(10))

    with Timer() as bid_features_time:
        add_flatten_features(bid_flatten_df, features, n, prefix="bid_flatten")
    logger.info(f"Base bid features time: {bid_features_time.elapsed:.4f} sec")
    logger.info(features.sample(10))

    with Timer() as interaction_time:
        features[f"flatten_spread_{n}_mean"] = (
            features[f"ask_flatten_mean_{n}"] / features[f"bid_flatten_mean_{n}"] - 1
        )
        features[f"flatten_spread_{n}_median"] = (
            features[f"ask_flatten_median_{n}"] / features[f"bid_flatten_median_{n}"] - 1
        )

        features[f"wap_flatten_{n}"] = (
            (features[f"ask_flatten_mean_{n}"] * features[f"bid_flatten_len_{n}"])
            + (features[f"bid_flatten_mean_{n}"] * features[f"ask_flatten_len_{n}"])
        ) / (features[f"ask_flatten_len_{n}"] + features[f"bid_flatten_len_{n}"])
    logger.info(f"Feature interaction time: {interaction_time.elapsed:.4f} sec")
    return ask_flatten_df, bid_flatten_df, features


def main():
    n_global = None
    data = read_data(n_global)
    os.makedirs("artefacts", exist_ok=True)
    for n_per_row in (5, 15, 50, 100):
        ask_flatten_df, bid_flatten_df, features = extract_flatten_features(data, n=n_per_row, fillna=False)
        features.to_pickle(f"artefacts/features_{n_per_row}.pkl")
        ask_flatten_df.to_pickle(f"artefacts/ask_flatten_{n_per_row}.pkl")
        bid_flatten_df.to_pickle(f"artefacts/bid_flatten_{n_per_row}.pkl")
        features.to_csv(f"artefacts/features_{n_per_row}.csv")


if __name__ == "__main__":
    main()