import pathlib
from pathlib import Path

import numba
import torch
from nleval import logger


def normalize_path(homedir: str) -> Path:
    path = pathlib.Path(homedir).expanduser().resolve()
    logger.info(f"Normlized path: {path}")
    return path


def get_data_dir(homedir: Path) -> Path:
    return homedir / "datasets"


def get_gene_list_path(homedir: Path) -> Path:
    return homedir / "genes.txt"


def get_device(device: str) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Automatically setting device to: {device!r}")
    return device


def get_num_workers(num_workers: int) -> int:
    default_num_workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
    if num_workers <= default_num_workers:
        return num_workers
    elif num_workers != 0:
        logger.warning(
            f"The specified {num_workers=} exceeds the maximum number of "
            f"available threads {default_num_workers}. "
            f"Setting to {default_num_workers} instead.",
        )
    return default_num_workers
