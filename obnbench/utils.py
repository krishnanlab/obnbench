import numba
from obnb import logger


def get_num_workers(num_workers: int) -> int:
    default_num_workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
    if 0 < num_workers <= default_num_workers:
        return num_workers
    elif num_workers > default_num_workers:
        logger.warning(
            f"The specified {num_workers=} exceeds the maximum number of "
            f"available threads {default_num_workers}. "
            f"Setting to {default_num_workers} instead.",
        )
    elif num_workers < 0:
        raise ValueError(f"num_workers should be a non-negative, got {num_workers}")
    logger.info(f"Setting number of workers to: {default_num_workers}")
    return default_num_workers
