Run the following script in the current experiment directory to submit jobs
for evaluating the methods.

```bash
sh submit_all.sh
```

Run the following script in the project root directory to aggregate results.

```bash
python aggregate_results.py -m tag -b experiments/node_feats -d
```

## Notes

- `[2023-03-31]` Fixed GIN in PyG 2.3 (use mean aggr)
- `[2023-03-29]` First run
    - Tested node features
        - `onehotlogdeg` [32, 64, 128]
        - `const` [1]
        - `random` [32, 64, 128]
        - `node2vec` [32, 64, 128]
        - `lappe` [32, 64, 128]
        - `rwse` [32, 64, 128]
    - Need to rerun out-of-time runs, i.e., the lappe 64 and 128 ones one
      BioGRID (potentially need more time for computing the lappe, but we don't
      need to worry about it atm as the lappe32 performances aren't looking
      really good)
