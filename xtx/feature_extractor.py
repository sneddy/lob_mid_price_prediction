from typing import List, Optional

import numpy as np
import pandas as pd

from xtx.modeling_utils import shrink_dtype


class FeatureExtractor:
    """Base feature factory"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._read_data()
        self.ask_rate_cols = [f"askRate{idx}" for idx in range(15)]
        self.ask_size_cols = [f"askSize{idx}" for idx in range(15)]
        self.bid_rate_cols = [f"bidRate{idx}" for idx in range(15)]
        self.bid_size_cols = [f"bidSize{idx}" for idx in range(15)]

        self.ask_rate = [self.data.loc[:, col] for col in self.ask_rate_cols]
        self.bid_rate = [self.data.loc[:, col] for col in self.bid_rate_cols]
        self.ask_size = [self.data.loc[:, col] for col in self.ask_size_cols]
        self.bid_size = [self.data.loc[:, col] for col in self.bid_size_cols]

    def calc_wap(self, order: int):
        return (self.bid_rate[order] * self.ask_size[order] + self.ask_rate[order] * self.bid_size[order]) / (
            self.ask_size[order] + self.bid_size[order]
        )

    def calc_volume_imbalance(self, order: int):
        return (self.bid_size[order] - self.ask_size[order]) / (self.ask_size[order] + self.bid_size[order])

    def calc_cum_volume_imbalance(self, order: int):
        cum_bid_size = self.data[self.bid_size_cols[:order]].sum(axis=1)
        cum_ask_size = self.data[self.ask_size_cols[:order]].sum(axis=1)
        return (cum_bid_size - cum_ask_size) / (cum_bid_size + cum_ask_size)

    def get_base_features(self, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract really reasonable features without any time dependensies
        Args:
            usecols (Optional[List[str]], optional): predefined subset of features to use.
                Defaults to None.
        Returns:
            pd.DataFrame: features
        """
        base_features = pd.DataFrame()
        base_features["ask_rate_0"] = self.ask_rate[0]
        # base_features['bid_rate_0'] = bid_rate[0]
        base_features["mid_price"] = (self.ask_rate[0] + self.bid_rate[0]) / 2
        base_features["mid_price_log"] = base_features["mid_price"].apply(np.log1p)

        # base_features['ask_size_0'] = self.ask_size[0].apply(np.log1p)
        # base_features['bid_size_0'] = self.bid_size[0].apply(np.log1p)

        # base_features['bid_ask_spread'] = self.ask_rate[0] / self.bid_rate[0] - 1
        base_features["ask_len"] = self.data[self.ask_size_cols].sum(axis=1)
        base_features["bid_len"] = self.data[self.bid_size_cols].sum(axis=1)
        base_features["wap0"] = self.calc_wap(0).astype(np.float32)
        # base_features['wap0_log'] = calc_wap(0).astype(np.float32).apply(np.log1p)
        base_features["wap1"] = self.calc_wap(1).astype(np.float32)
        base_features["len_ratio"] = base_features["ask_len"] / base_features["bid_len"]
        base_features["volume_imbalance"] = self.calc_volume_imbalance(0)
        base_features["volume_imbalance_1"] = self.calc_volume_imbalance(1)
        base_features["volume_imbalance_2"] = self.calc_volume_imbalance(2)

        # # how many ask size columns changed?
        ask_size_diff = self.data[self.ask_size_cols].diff(1)
        ask_size_diff.columns = np.arange(15)
        base_features["increased_ask_counts"] = (ask_size_diff > 0).sum(axis=1)
        base_features["increased_ask_rank"] = (ask_size_diff > 0).idxmax(axis=1)
        base_features.loc[base_features["increased_ask_counts"] == 0, "increased_ask_rank"] = 15
        base_features["decreased_ask_counts"] = (ask_size_diff < 0).sum(axis=1)
        base_features["decreased_ask_rank"] = (ask_size_diff < 0).idxmax(axis=1)
        base_features.loc[base_features["decreased_ask_counts"] == 0, "decreased_ask_rank"] = 15

        bid_size_diff = self.data[self.bid_size_cols].diff(1)
        bid_size_diff.columns = np.arange(15)
        base_features["increased_bid_counts"] = (bid_size_diff > 0).sum(axis=1)
        base_features["increased_bid_rank"] = (bid_size_diff > 0).idxmax(axis=1)
        base_features.loc[base_features["increased_bid_counts"] == 0, "increased_bid_rank"] = 15
        base_features["decreased_bid_counts"] = (bid_size_diff < 0).sum(axis=1)
        base_features["decreased_bid_rank"] = (bid_size_diff < 0).idxmax(axis=1)
        base_features.loc[base_features["decreased_bid_counts"] == 0, "decreased_bid_rank"] = 15

        # base_features['increased_askbid_counts'] = (data[ask_size_cols + bid_size_cols].diff(1) > 0).sum(axis=1)
        # base_features['decreased_askbid_counts'] = (data[ask_size_cols + bid_size_cols].diff(1) < 0).sum(axis=1)

        # shrink_dtype(base_features)
        if usecols is None:
            return base_features

        return base_features[usecols]

    def _read_data(self):
        extension = self.data_path.split(".")[-1]
        if extension == "csv":
            data = pd.read_csv(self.data_path)
        elif extension == "pkl":
            data = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"Incorrect extension of {self.data_path}. Expected .csv or .pkl")
        return data.fillna(0)
