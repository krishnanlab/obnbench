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
      - [x] Fix `n2v` results.
      - [x] Finish running `DisGeNET_BEFREE` experiments using `ADJ-LOGREG`.
      - [ ] Tune `GAT` to get more reasonable performance (now it is the worst
        performing GNN).

## Logs notes
- [logs/get_data_2023-02-16.txt](logs/get_data_2023-02-16.txt)
  Update experiment setting: add `ConsensusPathDB` network in addition to
  `HumanNet`. The goal is to see when using `ConsensusPathDB` as the underlying
  network data, if the network-based machine learning methods can better
  capture `curated` annotations than the `text-mined` annotations.
- [logs/aggregate_results_2023-02-16_2.txt](logs/aggregate_results_2023-02-16_1.txt)
  Aggregate results after the reran `DisGeNET_BEFREE` finished (~4.5 hours).
- [logs/aggregate_results_2023-02-16_1.txt](logs/aggregate_results_2023-02-16_1.txt)
  Aggregate results after the patched `n2v` runs are finished (previously
  bugged due to incorrectly set `num_workers` in `w2v`).
- [logs/aggregate_results_2023-02-16.txt](logs/aggregate_results_2023-02-16.txt)
  Aggregate results for experiments ran on 2023-02-14
- [logs/get_data_2023-02-14_1.txt](logs/get_data_2023-02-14_1.txt)
  Changing default network to `HumanNet` to be consistent with the `nleval`
  manuscript main results. Another major reason for not using `STRING` in the
  current state is due to computational cost (`STRING` is much denser than
  `HumanNet` and hance takes longer to run for some GNN methods).
- [logs/get_data_2023-02-14.txt](logs/get_data_2023-02-14.txt)
  Logs for preparing datasets using the `STRING` network.
