import os

import numpy as np
import pandas as pd

import xtx.utils.dev_utils as dev_utils
from xtx.modeling.evaluation import get_mse_and_corr_scores

logger = dev_utils.init_logger("logging/scoring.log")

if __name__ == "__main__":
    predictions_dir = "predictions/debug"
    ensemble_predictions = np.load(os.path.join(predictions_dir, "ensemble.npy"))
    stacking_predictions = np.load(os.path.join(predictions_dir, "stacking.npy"))

    target = pd.read_csv("data/simulation/test.csv")["y"]
    get_mse_and_corr_scores(target, ensemble_predictions, verbose=True, prefix="Hold Out simulation by ensembling")
    get_mse_and_corr_scores(target, stacking_predictions, verbose=True, prefix="Hold Out simulation by stacking")
