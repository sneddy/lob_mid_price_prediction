import os

import numpy as np

prediction_dirs = [
    "predictions/auto_features_fold0"
    "predictions/auto_features_fold1"
    "predictions/auto_features_fold2"
    "predictions/auto_features_fold3"
    "predictions/auto_features_fold4"
]

averaged_prediction = None
for prediction_dir in prediction_dirs:
    prediction_fpath = os.path.join(prediction_dir, "stacking.npy")
    current_prediction = np.load(prediction_fpath)
    if averaged_prediction is None:
        averaged_prediction = current_prediction / len(prediction_dirs)
    else:
        averaged_prediction += current_prediction / len(prediction_dirs)

np.save("predictions/averaged_predictions", averaged_prediction)
