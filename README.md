# nleval-benchmark

This is a benchmarking repository accompanying the [`nleval`](https://github.com/krishnanlab/NetworkLearningEval) Python package.

## Getting started

### Set up environment

Use the setup script provided to set up the `nleval` environment

```bash
git clone --depth 1 --branch v0.1.0-dev5 https://github.com/krishnanlab/NetworkLearningEval && cd NetworkLearningEval
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
