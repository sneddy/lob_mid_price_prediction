import numpy as np
import pandas as pd

import xtx.utils.dev_utils as dev_utils
from xtx.modeling.evaluation import get_mse_and_corr_scores
from xtx.modeling.runners import CrossValRunner
from xtx.modeling.time_folds import TimeFolds


def score_cols(fold_processor, runner, usecols, prefix=None):
    model = runner.init_model()
    model.fit(fold_processor.train_data[:, usecols], fold_processor.train_target)
    val_predicted = model.predict(fold_processor.valid_data[:, usecols])
    return get_mse_and_corr_scores(fold_processor.valid_target, val_predicted, verbose=True, prefix=prefix)


def select_features_for_fold(runner, train_features, fold_id):
    fold_processor = runner.preprocessor.prepare_fold(fold_id)
    n_features = train_features.shape[1]
    dev_utils.init_seed(42 + fold_id)
    best_usecols = np.random.choice(n_features, n_features // 2, replace=False).tolist()
    best_mse, best_corr = score_cols(fold_processor, runner, usecols=best_usecols)

    not_improved_iter = 0
    for current_iter in range(100000):
        if current_iter % 2 == 0:
            removing_col = np.random.choice(best_usecols)
            use_cols = [col for col in best_usecols if col != removing_col]
            prefix = f"---{removing_col}"
        else:
            adding_col = np.random.choice([idx for idx in range(n_features) if idx not in best_usecols])
            use_cols = best_usecols + [adding_col]
            prefix = f"+++{adding_col}"
        current_mse, current_corr = score_cols(fold_processor, runner, usecols=use_cols, prefix=prefix)
        if current_mse < best_mse:
            best_usecols = use_cols
            best_mse = current_mse
            not_improved_iter = 0
            if current_iter % 2 == 0:
                print(f"SUCCESSFULL REMOVE OF {train_features.columns[removing_col]}. Len: {len(use_cols)}")
            else:
                print(f"SUCCESSFULL ADDING OF {train_features.columns[adding_col]}. Len: {len(use_cols)}")
                print(f"Last added cols: {[train_features.columns[col] for col in best_usecols[-5:]]}")
        else:
            not_improved_iter += 1
            print(f"Not improved: {not_improved_iter}")
        if not_improved_iter > 20:
            break
    return best_usecols


def main(experiment):
    model_zoo = dev_utils.load_yaml(experiment["model_zoo"])
    # model_config = model_zoo['train_zoo']['default_ridge']
    model_config = model_zoo["selection_ridge"]
    train_features = pd.read_pickle(experiment["cached_features"])
    target = pd.read_csv(experiment["train_data_path"], usecols=["y"]).y

    for fold_id in range(5):
        time_folds = TimeFolds(**experiment["TimeFolds"])
        print(experiment["TimeFolds"])
        time_folds.fit(train_features, target)

        runner = CrossValRunner(time_folds, **model_config)
        best_usecols = select_features_for_fold(runner, train_features, fold_id)
        best_usenames = train_features.columns[best_usecols]
        with open(f"usecols/usecols_fold_{fold_id}.txt", "w") as f:
            f.writelines("\n".join(best_usenames))


if __name__ == "__main__":
    experiment = dev_utils.load_yaml("configs/experiment.yaml")
    main(experiment)
