import numpy as np
import pandas as pd

from xtx.modeling.evaluation import get_mse_and_corr_scores

if __name__ == "__main__":
    predictions = np.load("predictions/debug_ensemble.npy")
    target = pd.read_csv("data/simulation/test.csv")["y"]
    get_mse_and_corr_scores(target, predictions, verbose=True, prefix="Hold Out simulation")
