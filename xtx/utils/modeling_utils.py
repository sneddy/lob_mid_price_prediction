import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def draw_lgbm_feature_importance(clf, columns: List[str]):
    warnings.simplefilter(action="ignore", category=FutureWarning)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, columns)), columns=["Value", "Feature"])

    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False),
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.show()


def shrink_dtype(df: pd.DataFrame):
    """Float64 -> Float32, inplace"""
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
