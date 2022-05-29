import os
from typing import Any, Dict

import pandas as pd
from contexttimer import Timer

import xtx.utils.dev_utils as dev_utils
from xtx.features.feature_extractor import FeatureExtractor
from xtx.modeling.runners import CrossValClassificationRunner, CrossValRunner

logger = dev_utils.init_logger("logging/factory.log")


def build_runners(
    time_folds,
    model_congigs: Dict[str, Dict[str, Any]],
    runners_dir: str,
    use_cache: bool = True,
    regression: bool = True,
) -> Dict[str, CrossValRunner]:
    runners = {}

    for name, model_config in model_congigs.items():
        cached_runner_dir = os.path.join(runners_dir, name)
        if use_cache and CrossValRunner.cache_exists(cached_runner_dir):
            runners[name] = CrossValRunner.load(cached_runner_dir)
            logger.info(runners[name].report)
        else:
            if regression:
                current_runner = CrossValRunner(time_folds, **model_config)
            else:
                current_runner = CrossValClassificationRunner(n_classes=3, time_folds=time_folds, **model_config)
            current_runner.fit(verbose=True)
            runners[name] = current_runner
            current_runner.save(cached_runner_dir)
    return runners


def load_features(data_path: str, features_path: str, fake_target: int = None, use_cache=False):
    feature_extractor = FeatureExtractor(data_path)
    data = feature_extractor.data

    if use_cache and features_path is not None and os.path.exists(features_path):
        logger.info(f"Loading cached features from {features_path}")
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        merged_features = pd.read_pickle(features_path)
        # return merged_features, data.y
    else:
        with Timer() as base_features_time:
            base_features = feature_extractor.get_base_features()
        logger.info(f"Base features extraction time: {base_features_time.elapsed:.1f} sec")
        logger.info(f"Extracted base features: \n{base_features.columns.tolist()}")

        with Timer() as topk_features_time:
            topk_features = feature_extractor.get_topk_features()
        logger.info(f"Topk features extraction time: {topk_features_time.elapsed:.1f} sec")
        logger.info(f"Extracted topk features: \n{topk_features.columns.tolist()}")

        with Timer() as time_features_time:
            time_features = feature_extractor.get_time_base_features(base_features)
        logger.info(f"Topk features extraction time: {time_features_time.elapsed:.1f} sec")
        logger.info(f"Extracted topk features: \n{time_features.columns.tolist()}")

        merged_features = pd.concat((base_features, topk_features, time_features), axis=1)

        logger.info(f"cache features into {features_path}")
        merged_features.to_pickle(features_path)

    target = data.y if "y" in data.columns else None
    if fake_target is not None:
        logger.info(f"Making fake target: mid_price_diff_{fake_target}")
        target = feature_extractor.get_fake_target(fake_target).iloc[fake_target:]
        merged_features = merged_features.iloc[fake_target:, :]
    return merged_features, target
