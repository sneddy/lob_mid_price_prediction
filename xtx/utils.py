import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch


def init_seed(SEED=42):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def init_logger(log_path: str):
    formatter = logging.Formatter(
        "\n%(asctime)s\t%(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_path.name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
