# nleval-benchmark

This is a benchmarking repository accompanying the [`nleval`](https://github.com/krishnanlab/NetworkLearningEval) Python package.

## Getting started

### Set up environment

Use the setup script provided to set up the `nleval` environment

```bash
git clone --depth 1 --branch v0.1.0-dev4 https://github.com/krishnanlab/NetworkLearningEval && cd NetworkLearningEval
source install.sh cu102  # other options are [cpu,cu113]
```

Install an additional library [`PecanPy`](https://github.com/krishnanlab/PecanPy) for generating node2vec embeddings

```bash
pip install pecanpy==2.0.8
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
python main.py --network_name BioGRID --label_name DisGeNet --model_name GCN
```

The result file will be saved under the `results/` directory as a JSON file.
The file name for this particular run will be `biogrid_disgenet_gcn_0.json`.

All available options are

* `--network_name: [BioGRID,HumanNet,STRING]`
* `--label_name: [DisGeNet,GOBP]`
* `--model_name: [ADJ-LogReg,ADJ-SVM,N2V-LogReg,N2V-SVM,GCN,GIN,GAT,GraphSAGE]`

### Submit batch jobs

To run experiments for all combinations of network, label, and model, with ten repetitions, use the run script

```bash
cd run
sh submit_all.sh
```

which submit all experiements as SLURM jobs to run.


## Stats

### Label stats

**Note:** To make the comparison against label-rate from other benchmarks that are in multi-class settings instead of multi-label, we consider the notion of *effective class*, where each unique combinations of labels is considered as a "class".

#### DisGeNet

Total number of (effective) classes: 123 (2,947)

| | Label rate | Number of examples per class (avg) | Number of examples per class (std) | Effective number of examples per class (avg) | Effective number of examples per class (std)|
| ---------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Train | 0.210414 | 139.723577 | 87.124087 | 1.569231 | 3.607748 |
| Validation | 0.070138 | 43.560976 | 28.280001 | 1.637956 | 2.534618 |
| Testest | 0.070201 | 28.065041 | 17.743310 | 2.359244 | 4.862655 |

#### GOBP

Total number of (effective) classes: 203 (3,969)

| | Label rate | Number of examples per class (avg) | Number of examples per class (std) | Effective number of examples per class (avg) | Effective number of examples per class (std)|
| ---------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| train | 0.274739 | 85.842365 | 31.674838 | 1.500512 | 2.258730 |
| val | 0.091580 | 19.881773 | 7.133202 | 1.622370 | 1.594832 |
| test | 0.091580 | 18.024631 | 8.608301 | 1.873402 | 2.260405 |
