import glob

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.argument("name_tag", type=str)
def main(name_tag):
    res = []
    pbar = tqdm(glob.glob(f"results/{name_tag}/*/*/final_scores.csv"))
    for path in pbar:
        pbar.write(f"Loading data from {path}")
        _, _, name, run, _ = path.split("/")
        terms = name.split("-")
        if len(terms) == 3:
            network, label, method = terms
        elif len(terms) == 4:
            network, label, method, _ = terms
        else:
            raise ValueError(f"Unknwon terms length {len(terms)}: {terms=}")

        score = (
            pd
            .read_csv(path, index_col=0)
            .query("split == 'test' & score_type == 'APOP'")["score_value"]
            .mean()
        )
        res.append({
            "name": name,
            "network": network,
            "label": label,
            "method": method,
            "run": int(run.split("_")[1]),
            "score": score,
        })

    df_raw = pd.DataFrame(res)
    df = (
        df_raw
        .drop(columns=["name", "run"])
        .groupby(["network", "label", "method"])
        .agg(lambda x: f"{x.mean():.3f} ± {x.std():.3f}")
        .reset_index()
        .pivot_table(
            columns=["network", "label"],
            index="method",
            values="score",
            aggfunc=lambda x: x,
        )
    )
    print(f"\n{df}\n{df.to_latex()}\n")

    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    main()
