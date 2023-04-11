import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from glob import glob
from pprint import pformat
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm
from nleval.util.logger import get_logger

from main import ALL_METHODS


def parse_args() -> Tuple[argparse.Namespace, logging.Logger]:
    global logger

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--basedir", default=".", help="Base directory")
    parser.add_argument("-m", "--mode", required=True, choices=["main", "hp_tune", "tag"],
                        help="'main' and 'hp_tune' for main and hyperparameter tuning experiments.")
    parser.add_argument("-p", "--results_path", default="auto",
                        help="Path to the results directory, infer from 'mode' if set to 'auto'.")
    parser.add_argument("-n", "--dry_run", action="store_true",
                        help="Aggregate and print results, but do not save to disk.")
    parser.add_argument("-o", "--output_path", default="aggregated_results/")
    parser.add_argument("-d", "--suffix_date", action="store_true",
                        help="Add date suffix to file name if set.")
    parser.add_argument("-v", "--log_level", type=str, default="INFO")
    parser.add_argument("--methods", type=str, nargs="+", default=ALL_METHODS,
                        help="List of methods to consider when aggregating results.")

    # Parse arguments from command line and set up logger
    args = parser.parse_args()
    logger = get_logger(None, log_level=args.log_level)
    logger.info(f"Settings:\n{pformat(vars(args))}")

    return args


def _agg_main_results(
    results_path: str,
    target_methods: List[str],
) -> pd.DataFrame:
    df_lst = []
    target_methods_lower = list(map(str.lower, target_methods))
    for path in tqdm(glob(osp.join(results_path, "*.json"))):
        terms = osp.splitext(osp.split(path)[1])[0].split("_")  # network, label, method, runid

        # Some label names contains, e.g., disgenet_curated
        # These would then become [network, labelpart1, labelpart2, method, runid]
        # We want to combine the label parts into a single string again
        terms = [
            terms[0],
            "_".join(terms[1:-2]),
            *terms[-2:],
        ]

        if terms[2] not in target_methods_lower:
            logger.warning(f"Skipping {terms[2]}: {path}")

        df_lst.append(pd.read_json(path))
        df_lst[-1][["network", "label", "method", "runid"]] = terms
    return pd.concat(df_lst)


def _agg_hp_results(
    results_path: str,
    target_methods: List[str],
    target_file: str = "score.json",
) -> pd.DataFrame:
    df_lst = []
    for dir_, _, files in tqdm(list(os.walk(results_path))):
        if target_file in files:
            terms = dir_.split(osp.sep)[-4:]  # method, settings, dataset (netowkr-label), runid
            if terms[0] not in target_methods:
                logger.debug(f"Skipping {terms[0]}: {dir_}")
                continue

            path = osp.join(dir_, target_file)
            df_lst.append(pd.read_json(path))
            df_lst[-1][["method", "settings", "dataset", "runid"]] = terms
    return pd.concat(df_lst)


def _agg_tag_results(
    results_path: str,
    target_methods: List[str],
) -> pd.DataFrame:
    id_terms = ["network", "label", "method", "tag", "runid"]

    dfs = []
    pbar = tqdm(glob(osp.join(results_path, "*.json")))
    for path in pbar:
        pbar.set_description(f"Loading results from {path:<80}")
        res = pd.read_json(path)
        terms = path.split("/")[-1].split(".json")[0].split("_")
        for i, j in zip(id_terms, terms):
            res[i] = j
        dfs.append(res)
    return pd.concat(dfs)


def main():
    args = parse_args()
    dry_run = args.dry_run
    methods = args.methods
    mode = args.mode
    suffix_date = args.suffix_date
    basedir = args.basedir
    output_path = args.output_path
    results_path = args.results_path

    # Get aggregation function and results_path
    if mode == "main":
        agg_func = _agg_main_results
        inferred_path = "results"
    elif mode == "hp_tune":
        agg_func = _agg_hp_results
        inferred_path = "hp_tune_results"
    elif mode == "tag":
        agg_func = _agg_tag_results
        inferred_path = "results"
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    results_path = inferred_path if results_path == "auto" else results_path

    # Attach base directory to paths
    output_path = osp.join(basedir, output_path)
    results_path = osp.join(basedir, results_path)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # Aggregate results
    logger.info(f"Results will be aggregated from {results_path}")
    logger.info(f"Start aggregating results for methods: {methods}")
    results_df = agg_func(results_path, methods)
    logger.info(f"Aggregated results:\n{results_df}")

    # Construct output path
    suffix = f"_{datetime.now().strftime('%Y-%m-%d')}" if suffix_date else ""
    path = osp.join(output_path, f"{mode}_results{suffix}.csv")

    # Save or print results
    if dry_run:
        logger.info(f"Results will be saved to {path}")
    else:
        # Save single precision
        results_df.astype("float32", errors="ignore").to_csv(path, index=False)
        logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
