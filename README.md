# Benchmarking repository for the Open Biomedical Network Benchmark

This is a benchmarking repository accompanying the [`obnb`](https://github.com/krishnanlab/obnb) Python package.

## Getting started

### Set up environment

Use the setup script provided to set up the `obnb` environment

```bash
git clone git@github.com:krishnanlab/obnb.git && cd obnb
source install.sh cu117  # other options are [cpu,cu118]
pip install -e .[ext]  # install extension modules (PecanPy, GRAPE, ...)
```

Install additional dependencies, e.g.,

- [`Hydra`](https://github.com/facebookresearch/hydra) for managing experiments.
- [`Lightning`](https://lightning.ai/docs/pytorch/latest/) for organizing model training framework.
- [`WandB`](https://docs.wandb.ai/) for logging metrics.

```bash
pip install -r requirements_extra.txt
```

### Set up data

Run `get_data.py` to download and set up data for the experiments.
All data will be saved under the `datasets/` directory by default, and will take up approximately 6.1 GB of space.

```bash
python get_data.py
```

### Run experiments

After setting up the data, one can run a single experiment by specifying the choices of network, label, and model:

```bash
python main.py network=BioGRID label=DisGeNET model=GCN
```

For testing with hyperparameter tuning option:

```bash
python main.py network=BioGRID label=DisGeNET model=GCN out_dir=test gnn_params.epochs=5000 gnn_params.eval_steps=20 hp_tune=true
```

The result file will be saved under the `results/` directory as a JSON file.
The file name for this particular run will be `biogrid_disgenet_gcn_0.json`.

All available options are

* `network: [BioGRID,HumanNet,STRING]`
* `label: [DisGeNET,GOBP]`
* `model: [ADJ-LogReg,ADJ-SVM,N2V-LogReg,N2V-SVM,GCN,GIN,GAT,GraphSAGE]`

### Submit batch jobs

To run experiments for all combinations of network, label, and model, with ten repetitions, use the run script

```bash
cd run
sh submit_all.sh
# sh submit_all_hp_tune.sh  # hyper-parameter tuning
```

which submit all experiements as SLURM jobs to run.

### Results aggregation

Run the following commands to aggregate raw results into a csv file

```bash
python aggregate_results.py -m main
# python aggregate_results.py -m hp_tune  # aggregate hyperparameter tuning results
```

The aggregated results will be saved under `aggregated_results/` by default

## Stats

### Network stats

| Network | # nodes | # edges | Edge density |
| :-------: | -------: | -------: | -------: |
| BioGRID | 18,951 | 1,103,298 | 0.0031 |
| HumanNet | 17,211 | 847,104 | 0.0029 |
| STRING | 17,942 | 10,951,202 | 0.0340 |

### Label stats

**Note:** To make the comparison against label-rate from other benchmarks that are in multi-class settings instead of multi-label, we consider the notion of *effective class*, where each unique combinations of labels is considered as a "class".

#### DisGeNET

Total number of (effective) classes: 123 (2,947)

| | Label rate | Number of examples per class (avg) | Number of examples per class (std) | Effective number of examples per class (avg) | Effective number of examples per class (std)|
| ---------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Train | 0.210414 | 139.723577 | 87.124087 | 1.142178 | 3.155479 |
| Validation | 0.070138 | 43.560976 | 28.280001 | 0.380726 | 1.403474 |
| Test | 0.070201 | 28.065041 | 17.743310 | 0.381065 | 2.136587 |

#### GOBP

Total number of (effective) classes: 203 (3,969)

| | Label rate | Number of examples per class (avg) | Number of examples per class (std) | Effective number of examples per class (avg) | Effective number of examples per class (std)|
| ---------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Train | 0.274739 | 85.842365 | 31.674838 | 1.107332 | 2.049174 |
| Validation | 0.091580 | 19.881773 | 7.133202 | 0.369111 | 1.020112 |
| Test | 0.091580 | 18.024631 | 8.608301 | 0.369111 | 1.249263 |
