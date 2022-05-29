from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats
from pandarallel import pandarallel

import xtx.features.flatten_tools as flatten_tools
from xtx.utils.modeling_utils import shrink_dtype

pandarallel.initialize(progress_bar=True)


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

    def rate_moda(self, is_ask: bool):
        vals = self.data.values[:, :30] if is_ask else self.data.values[:, 30:60]
        rate_idx = vals[:, 15:30].argmax(1)
        return vals[np.arange(vals.shape[0]), rate_idx]

    def get_fake_target(self, shift_size: int):
        """Fake target. Can include nan values in first shift_size positions"""
        return -(self.data["askRate0"] + self.data["bidRate0"]).diff(-shift_size)

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

        # difference with most frequent count for ask and bid (to remove)
        base_features["ask_rate_moda_spread"] = self.rate_moda(is_ask=True) - self.data.askRate0
        base_features["bid_rate_moda_spread"] = self.data.bidRate0 - self.rate_moda(is_ask=False)

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

        shrink_dtype(base_features)
        if usecols is None:
            return base_features

        return base_features[usecols]

    def get_time_base_features(self, base_features: pd.DataFrame, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        time_features = pd.DataFrame()
        # time_features["deal_flag"] = ((self.data.askRate0.diff() * self.data.bidRate0.diff()) > 0).astype(np.int8)
        for window in (3, 5, 10, 20, 40, 80):
            # time_features["increased_ask_counts"]
            # ask bid size changes
            ask_size_diff = self.data[self.ask_size_cols].diff(window)
            bid_size_diff = self.data[self.bid_size_cols].diff(window)

            time_features["increased_ask_counts"] = (ask_size_diff > 0).sum(axis=1)
            time_features["decreased_ask_counts"] = (ask_size_diff < 0).sum(axis=1)

            # time_features[f'increased_ask_counts_{window}_volume'] = (ask_size_diff * (ask_size_diff > 0)).sum(1)
            # time_features[f'decreased_ask_counts_{window}_volume'] = (-ask_size_diff * (ask_size_diff < 0)).sum(1)

            time_features["increased_bid_counts"] = (bid_size_diff > 0).sum(axis=1)
            time_features["decreased_bid_counts"] = (bid_size_diff < 0).sum(axis=1)

            time_features[f"volume_imbalance_{window}"] = base_features["volume_imbalance"].diff(window)
            time_features[f"mid_price_log_{window}"] = base_features["mid_price_log"].diff(window)

        for window in (40, 80):
            time_features[f"mid_price_log_max_diff_{window}"] = (
                base_features["mid_price_log"].rolling(window).max() - base_features["mid_price_log"]
            )
            # time_features[f'mid_price_log_std_{window}'] = \
            # base_features[f'mid_price_log'].diff(window).rolling(window).std()

        for window in (10, 20, 40, 80):
            time_features[f"wap0_{window}_mean"] = base_features["wap0"].rolling(window).mean()
            time_features[f"wap0_{window}_std"] = base_features["wap0"].rolling(window).std()  # overfit?
            time_features[f"wap0_{window}_max"] = base_features["wap0"].rolling(window).max()  # overfit?

            # time_features[f"wap1_{window}_mean"] = base_features["wap1"].rolling(window).mean()
            # time_features[f"wap1_{window}_std"] = base_features["wap1"].rolling(window).std()  # overfit?
            # time_features[f"wap1_{window}_max"] = base_features["wap1"].rolling(window).max()  # overfit?

            time_features[f"volume_imbalance_{window}_mean"] = base_features["volume_imbalance"].rolling(window).mean()
            time_features[f"volume_imbalance_{window}_max"] = base_features["volume_imbalance"].rolling(window).max()
            time_features[f"volume_imbalance_{window}_std"] = base_features["volume_imbalance"].rolling(window).std()
            time_features[f"volume_imbalance_{window}_skew"] = base_features["volume_imbalance"].rolling(window).skew()

            # time_features[f'volume_imbalance_{window}_iqr'] = \
            # base_features['volume_imbalance'].rolling(window).quantile(0.75) - \
            # time_features['volume_imbalance'].rolling(window).quantile(0.25)
            time_features[f"len_ratio_{window}_mean"] = base_features["len_ratio"].rolling(window).mean()
            time_features[f"len_ratio_{window}_std"] = base_features["len_ratio"].rolling(window).std()
        shrink_dtype(time_features)
        if usecols is None:
            return time_features
        return time_features[usecols]

    def get_topk_features(self):
        """Build small subset of topk features"""
        features = pd.DataFrame()
        ask_flatten_df_5 = self.data.parallel_apply(lambda x: flatten_tools.ask_flatten(x, n=5), axis=1)
        bid_flatten_df_5 = self.data.parallel_apply(lambda x: flatten_tools.ask_flatten(x, n=5), axis=1)
        ask_flatten_df_50 = self.data.parallel_apply(lambda x: flatten_tools.ask_flatten(x, n=50), axis=1)
        bid_flatten_df_50 = self.data.parallel_apply(lambda x: flatten_tools.ask_flatten(x, n=50), axis=1)
        features["bid_flatten_mean_5"] = bid_flatten_df_5.parallel_apply(np.mean)
        features["ask_flatten_mean_5"] = ask_flatten_df_5.parallel_apply(np.mean)
        features["bid_flatten_mean_50"] = bid_flatten_df_50.parallel_apply(np.mean)
        features["ask_flatten_mean_50"] = ask_flatten_df_50.parallel_apply(np.mean)
        features["ask_flatten_skew_50"] = ask_flatten_df_50.parallel_apply(scipy.stats.skew)
        features["ask_flatten_iqr_50"] = ask_flatten_df_50.parallel_apply(scipy.stats.iqr)
        features["bid_flatten_kurtosis_50"] = bid_flatten_df_50.parallel_apply(scipy.stats.kurtosis)
        features["bid_flatten_std_50"] = bid_flatten_df_50.parallel_apply(np.std)
        return features

    def _read_data(self):
        extension = self.data_path.split(".")[-1]
        if extension == "csv":
            data = pd.read_csv(self.data_path)
        elif extension == "pkl":
            data = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"Incorrect extension of {self.data_path}. Expected .csv or .pkl")
        print(f"Loaded data from {self.data_path} with shape: {data.shape}")
        return data.fillna(0)
