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

- `[2023-03-07]` Add `HumanNet-FN` and `HumanNet-CC` networks
    - TODOs
        - [ ] Rerun out-of-time runs (DisGeNET x Adj-LogReg). We can skip the
          analysis for DisGeNET x Adj-LogReg.
            - [ ] consensuspathdb-disgenet-adj-logreg
            - [ ] humannet-cc-disgenet-adj-logreg
            - [ ] humannet-cc-disgenet-adj-logreg
            - [ ] humannet-disgenet-adj-logreg
            - [ ] humannet-disgenet-adj-logreg
            - [ ] humannet-disgenet-adj-logreg
            - [ ] humannet-disgenet-adj-logreg
            - [ ] humannet-fn-disgenet-adj-logreg
            - [ ] humannet-fn-disgenet-adj-logreg

- `[2023-02-17]` Add `ConsensusPathDB` network to the experiment.
    - TODOs
        - [x] Rerun the following due to time out (allocate 6hrs)
            - `consensuspathdb-diseases_textminingfiltered-adj-logreg`
            - `consensuspathdb-disgenet_befree-adj-logreg`
            - `humannet-diseases_textminingfiltered-adj-logreg`
        - [x] The following ran out of time again, might need to allocate 24hrs
            - `consensuspathdb-diseases_textminingfiltered-adj-logreg`
            - `consensuspathdb-disgenet_befree-adj-logreg`
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
- [logs/get_data_2023-03-04.txt](logs/get_data_2023-03-04.txt)
  Add `HumanNet-FN` and `HumanNet-CC` networks in addition to the fully
  integrated `HumanNet-XC` (switched to `latest` version, but only updated
  data related to `HumanNet`).
- [logs/aggregate_results_2023-02-27.txt](logs/aggregate_results_2023-02-27.txt)
  Fix out-of-time runs again (need to allocate 24 hours)
- [logs/aggregate_results_2023-02-21.txt](logs/aggregate_results_2023-02-21.txt)
  Fix out-of-time runs (`consensuspathdb-diseases_textminingfiltered-adj-logreg`,
  `consensuspathdb-disgenet_befree-adj-logreg`,
  `humannet-diseases_textminingfiltered-adj-logreg`)
- [logs/aggregate_results_2023-02-16.txt](logs/aggregate_results_2023-02-16.txt)
  Update results after adding `ConsensusPathDB` network
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