import functools
from typing import List

import pandas as pd


def flatten(row, n=-1, ask=True, fillna=False) -> List[float]:
    """
    Args:
        row: row in dataframe
        ask: if True returns n smallest ask rates as a List.
            Otherwise returns n largest bid rates
        n (int): number of rates to return. If n==-1 then n not limited
    """
    assert fillna or n != -1
    prefix = "ask" if ask else "bid"

    outputs = []
    for idx in range(15):
        col_rate = row[f"{prefix}Rate{idx}"]
        col_size = row[f"{prefix}Size{idx}"].astype(int)
        n_rest = n - len(outputs)
        if n == -1 or col_size < n_rest:
            outputs.extend([col_rate] * col_size)
        else:
            outputs.extend([col_rate] * n_rest)
            break
    if fillna:
        n_rest = n - len(outputs)
        rest_values = outputs[-1] + 1 if ask else outputs[-1] - 1
        outputs.extend([rest_values] * n_rest)
    return outputs


def flatten_series(row, n: int, ask=True) -> pd.Series:
    assert n != -1, ValueError()

    prefix = "ask" if ask else "bid"
    outputs = flatten(row=row, n=n, ask=ask, fillna=True)
    return pd.Series(outputs, index=[f"{prefix}#{i}" for i in range(n)])


ask_flatten = functools.partial(flatten, ask=True)
bid_flatten = functools.partial(flatten, ask=False)
ask_flatten_series = functools.partial(flatten_series, ask=True)
bid_flatten_series = functools.partial(flatten_series, ask=False)
