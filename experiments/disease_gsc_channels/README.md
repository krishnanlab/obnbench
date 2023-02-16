## Overview


## Scripts

Run the following script at the project root directory to prepare the datasets.

```bash
python get_data.py +experiment=disease_gsc_channels
```

Run the following script in the current experiment directory to submit jobs
for evaluating the methods.

```bash
sh submit_all.sh
```

Run the following script at the project root directory to aggregate the (main)
results.

```bash
python aggregate_results.py -d -m main -b experiments/disease_gsc_channels
```

## Notes

- `[2023-02-14]` Investigate the performance difference between different
  annotation sources, e.g., text-mining vs. curated vs. experiments, across
  all network based machine learning methods.
    - Summary: Difficulties for the network based machine learning methods to
      capture different annotation sources can be roughly ordered as follow:
      *experiments* > *inferred* > *curated* > *text-mined*
    - TODOs
      - [ ] Fix `n2v` results.
      - [ ] Finish running `DisGeNET_BEFREE` experiments using `ADJ-LOGREG`.
      - [ ] Tune `GAT` to get more reasonable performance (now it is the worst
        performing GNN).
