import os
from datetime import datetime
from glob import glob

import pandas as pd
from tqdm import tqdm

ID_TERMS = ["network", "label", "method", "nodefeat", "runid"]
OUT_DIR = "aggregated_results"


def main():
    dfs = []
    pbar = tqdm(glob("results/*.json"))
    for path in pbar:
        pbar.set_description(f"Loading results from {path:<70}")
        res = pd.read_json(path)
        terms = path.split("/")[-1].split(".json")[0].split("_")
        for i, j in zip(ID_TERMS, terms):
            res[i] = j
        dfs.append(res)
    df = pd.concat(dfs)

    os.makedirs(OUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(OUT_DIR, f"{date_str}.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
