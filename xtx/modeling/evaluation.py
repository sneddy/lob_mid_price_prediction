from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def get_mse_and_corr_scores(gt, predicted, verbose=False, prefix=None):
    prefix = "" if prefix is None else prefix
    mse_score = mean_squared_error(gt, predicted)
    corr_score = np.corrcoef(gt, predicted)[0, 1]
    if verbose:
        print(f"{prefix} MSE score: {mse_score:.4f}; {prefix} Corr score: {corr_score:.4f}")
    return mse_score, corr_score


class CrossValReport:
    """Calculate and store relevant metrics and statistics"""

    def __init__(self, n_folds: int, test_eval: bool = True):
        self.n_folds = n_folds
        self.test_eval = test_eval

        self.test_target: np.ndarray = None
        self.averaged_test_predicted: np.ndarray = None
        self.val_mse_scores: List[float] = []
        self.val_corr_scores: List[float] = []
        self.test_mse_scores: List[float] = []
        self.test_corr_scores: List[float] = []

    def update_val(self, val_target: np.ndarray, val_predicted: np.ndarray):
        mse_score, corr_score = get_mse_and_corr_scores(val_target, val_predicted)
        self.val_corr_scores.append(corr_score)
        self.val_mse_scores.append(mse_score)

    def update_test(self, test_target: np.ndarray, test_predicted: np.ndarray):
        if self.test_target is None:
            self.test_target = test_target
        else:
            assert (self.test_target == test_target).all(), ValueError("Got different test target by folds")
        if self.averaged_test_predicted is None:
            self.averaged_test_predicted = test_predicted / self.n_folds
        else:
            self.averaged_test_predicted += test_predicted / self.n_folds

        mse_score, corr_score = get_mse_and_corr_scores(test_target, test_predicted)
        self.test_corr_scores.append(corr_score)
        self.test_mse_scores.append(mse_score)

    @property
    def val_mse_mean(self) -> float:
        return np.mean(self.val_mse_scores)

    @property
    def val_corr_mean(self) -> float:
        return np.mean(self.val_corr_scores)

    @property
    def test_mse_mean(self) -> float:
        if self.test_eval:
            return np.mean(self.test_mse_scores)

    @property
    def test_corr_mean(self) -> float:
        if self.test_eval:
            return np.mean(self.test_corr_scores)

    @property
    def val_mse_conf(self) -> Tuple[float, float]:
        sigma = np.std(self.val_mse_scores)
        return self.val_mse_mean - 2 * sigma, self.val_mse_mean + 2 * sigma

    @property
    def val_corr_conf(self) -> Tuple[float, float]:
        sigma = np.std(self.val_corr_scores)
        return self.val_corr_mean - 2 * sigma, self.val_corr_mean + 2 * sigma

    @property
    def test_mse_conf(self) -> Tuple[float, float]:
        if self.test_eval:
            sigma = np.std(self.test_mse_scores)
            return self.test_mse_mean - 2 * sigma, self.test_mse_mean + 2 * sigma

    @property
    def test_corr_conf(self) -> Tuple[float, float]:
        if self.test_eval:
            sigma = np.std(self.test_corr_scores)
            return self.test_corr_mean - 2 * sigma, self.test_corr_mean + 2 * sigma

    @property
    def test_averaged_mse(self) -> float:
        if self.test_eval:
            return mean_squared_error(self.test_target, self.averaged_test_predicted)

    @property
    def test_averaged_corr(self) -> float:
        if self.test_eval:
            return np.corrcoef(self.test_target, self.averaged_test_predicted)[0, 1]

    @property
    def folds_report(self) -> pd.DataFrame:
        if self.test_eval:
            df_dict = {
                ("val", "mse"): self.val_mse_scores,
                ("val", "corr"): self.val_corr_scores,
                ("test", "mse"): self.test_mse_scores,
                ("test", "corr"): self.test_corr_scores,
            }
        else:
            df_dict = {("val_mse"): self.val_mse_scores, ("val_corr"): self.val_corr_scores}
        df = pd.DataFrame.from_dict(df_dict).T.reset_index()
        index_cols = ["dataset", "metric_name"] if self.test_eval else ["metric_name"]
        df.columns = index_cols + [f"fold_{idx}" for idx in range(self.n_folds)]
        return df.round(3)

    def __repr__(self) -> str:
        val_repr = f"""\n \
            \tVal  corr score averaged: {self.val_corr_mean:.3f}
            \tVal   MSE score averaged: {self.val_mse_mean:.3f}
        """
        test_repr = f"""\n \
            \tTest corr score averaged: {self.test_corr_mean:.3f}
            \tTest  MSE score averaged: {self.test_mse_mean:.3f}
            ------------------------------------------------------------------
            \tTest  MSE predicts averaged: {self.test_averaged_mse:.3f}
            \tTest corr predicts averaged: {self.test_averaged_corr:.3f}"""
        aggregated_repr = val_repr + test_repr if self.test_eval else val_repr
        return self.folds_report.to_markdown() + aggregated_repr

    def save(self, report_path: str):
        with open(report_path, "w") as output_stream:
            output_stream.writelines(self.__repr__())
