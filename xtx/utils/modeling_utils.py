import os
import random
import warnings
from typing import List

import IPython.core.display as display
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


def draw_moment_book(row: pd.Series, radius: int = 15, horizontal_view=True) -> pd.DataFrame:
    row_dict = row.to_dict()
    ask_book = {
        row_dict[f"askRate{idx}"]: row_dict[f"askSize{idx}"] for idx in range(radius) if row_dict[f"askSize{idx}"] != 0
    }
    bid_book = {
        row_dict[f"bidRate{idx}"]: row_dict[f"bidSize{idx}"] for idx in range(radius) if row_dict[f"bidSize{idx}"] != 0
    }
    min_ask_price, max_ask_price = min(ask_book.keys()), max(ask_book.keys())
    ask_price_grid = np.linspace(min_ask_price, max_ask_price, int((max_ask_price - min_ask_price) * 2) + 1).tolist()
    min_bid_price, max_bid_price = min(bid_book.keys()), max(bid_book.keys())
    bid_price_grid = np.linspace(min_bid_price, max_bid_price, int((max_bid_price - min_bid_price) * 2) + 1).tolist()
    book = []
    for price in bid_price_grid + ask_price_grid:
        record = {"bid#": int(bid_book.get(price, 0)), "rate": price, "ask#": int(ask_book.get(price, 0))}
        book.append(record)

    book_df = pd.DataFrame(book).sort_values("rate", ascending=False).set_index("rate").T
    book_df_style = book_df.style
    book_df_style = book_df_style.set_properties(**{"background-color": "blue"}, subset=bid_price_grid)
    book_df_style = book_df_style.set_properties(**{"background-color": "green"}, subset=ask_price_grid)

    book_df_style.columns = [f"{price:.1}" for price in book_df_style.columns]
    if not horizontal_view:
        book_df_style = book_df_style.T

    display.display(book_df_style)


def set_all_seeds(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
