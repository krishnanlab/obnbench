# Benchmarking repository for the Open Biomedical Network Benchmark

This is a benchmarking repository accompanying the [`obnb`](https://github.com/krishnanlab/obnb) Python package.

## Set up environment

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

## Set up data (optional)

Run `get_data.py` to download and set up data for all the experiments.
Data will be saved under the `datasets/` directory by default, and will take up approximately 6 GB of space.

```bash
python get_data.py
```

This step is completely optional and directly runing the training script will work fine.
But runing `get_data.py` once before training prevents multiple parallel jobs doing the same data preprocessing
work if the processed data is not available yet.

## Run experiments

After setting up the data, one can run a single experiment by specifying the choices of network, label, and model:

```bash
python main.py dataset.network=BioGRID dataset.label=DisGeNET model=GCN
```

Check out the [`conf/model/`](conf/model) directory for all available model presets.
The main model presets are:

- `GCN`
- `GAT`
- `GCN+BoT`
- `GAT+BoT`
- `LogReg+Adj`
- `LogReg+Node2vec`
- `LogReg+Walklets`

### Run batch of parallel jobs

```bash
cd run

# GNN node feature ablation (example of runing GCN with node2vec features on BioGRID)
sh run_abl_gnn_feature.sh GCN BioGRID Node2vec

# C&S ablation (example of runing GCN with C&S post processing on BioGRID)
sh run_abl_cs.sh GCN BioGRID

# GNN label reuse ablation (example of runing GCN with label reuse on BioGRID)
sh run_abl_gnn_label.sh GCN BioGRID

# GNN label reuse with C&S ablation (example of runing GCN with label reuse with C&S on BioGRID)
sh run_abl_gnn_cs_label.sh GCN BioGRID

# GNN with bag of tricks, i.e., node2vec node feature + label reuse + C&S
sh run_gnn_bot.sh GCN BioGRID
```

To run all experiments presented in the paper (may take several days):

```bash
sh run_all.sh
```

### Tuning with W&B

First create a sweep agent, e.g., for BioGRID-DisGeNET-GCN:

```bash
wandb sweep conf/tune/BioGRID-DisGeNET-GCN.yaml
```

Then, follow the instruction from the command above to spawn sweep agents to automatically
tune the model configuration on a particular dataset.

## Results anallysis

To run the [notebooks](notebook), first download our benchmarking results (or you can rerun all the benchmarking
experiments yourself using our run scripts described above).

```bash
gdown --fuzzy -O results/main.csv.gz https://drive.google.com/file/d/1JUP3eTKuQnROMJ3xKpekNSOZJBPTmsuJ/view
```

## Data stats (`obnbdata-0.1.0`)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8045270.svg)](https://doi.org/10.5281/zenodo.8045270)

### Networks

| Network | Weighted | Num. nodes | Num. edges | Density | Category |
| :------ | :------: | ---------: | ---------: | ------: | -------: |
| [HumanBaseTopGlobal](https://humanbase.net/) | :white_check_mark: | 25,689 | 77,807,094 | 0.117908 | Large & Dense |
| [HuMAP](http://humap2.proteincomplexes.org/) | :white_check_mark: | 15,433 | 35,052,604 | 0.147180 | Large & Dense |
| [STRING](https://string-db.org/) | :white_check_mark: | 18,480 | 11,019,492 | 0.032269 | Large |
| [ConsensusPathDB](http://cpdb.molgen.mpg.de/) | :white_check_mark: | 17,735 | 10,611,416 | 0.033739 | Large |
| [FunCoup](https://funcoup.org/) | :white_check_mark: | 17,892 | 10,037,478 | 0.031357 | Large |
| [PCNet](https://www.ndexbio.org/viewer/networks/f93f402c-86d4-11e7-a10d-0ac135e8bacf) | :x: | 18,544 | 5,365,116 | 0.015603 | Large |
| [BioGRID](https://thebiogrid.org/) | :x: | 19,765 | 1,554,790 | 0.003980 | Medium |
| [HumanNet](https://staging2.inetbio.org/humannetv3/) | :white_check_mark: | 18,591 | 2,250,780 | 0.006513 | Medium |
| [HIPPIE](http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/) | :white_check_mark: | 19,338 | 1,542,044 | 0.004124 | Medium |
| [ComPPIHumanInt](https://comppi.linkgroup.hu/) | :white_check_mark: | 17,015 | 699,620 | 0.002417 | Medium |
| [OmniPath](https://omnipathdb.org/) | :x: | 16,325 | 289,134 | 0.001085 | Small |
| [ProteomeHD](https://www.ndexbio.org/viewer/networks/4cb4b0f3-83da-11e9-848d-0ac135e8bacf) | :x: | 2,471 | 125,172 | 0.020509 | Small |
| [HuRI](http://www.interactome-atlas.org/) | :x: | 8,100 | 103,188 | 0.001573 | Small |
| [BioPlex](https://bioplex.hms.harvard.edu/explorer/home) | :x: | 8,108 | 71,004 | 0.001080 | Small |
| [SIGNOR](https://signor.uniroma2.it/) | :x: | 5,291 | 28,676 | 0.001025 | Small |

### Labels

#### DISEASES

| Network | Num. tasks | Num. pos. avg. | Num. pos. std. | Num. pos. med. |
| :------ | ---------: | -------------: | -------------: | -------------: |
| BioGRID | 145 | 178.1 | 137.4 | 127.0 |
| BioPlex | 72 | 123.8 | 64.4 | 101.5 |
| ComPPIHumanInt | 145 | 174.6 | 134.5 | 125.0 |
| ConsensusPathDB | 144 | 177.4 | 137.5 | 126.0 |
| FunCoup | 145 | 177.1 | 135.1 | 127.0 |
| HIPPIE | 143 | 178.1 | 137.6 | 127.0 |
| HuMAP | 123 | 168.0 | 119.2 | 120.0 |
| HuRI | 50 | 130.3 | 56.7 | 112.5 |
| HumanBaseTopGlobal | 149 | 178.5 | 137.7 | 129.0 |
| HumanNet | 142 | 179.0 | 136.9 | 127.0 |
| OmniPath | 135 | 180.2 | 131.1 | 131.0 |
| PCNet | 143 | 171.8 | 130.6 | 122.0 |
| ProteomeHD | 15 | 76.9 | 22.4 | 70.0 |
| SIGNOR | 89 | 144.6 | 89.4 | 117.0 |
| STRING | 146 | 175.4 | 135.6 | 126.0 |

#### DisGeNET

| Network | Num. tasks | Num. pos. avg. | Num. pos. std. | Num. pos. med. |
| :------ | ---------: | -------------: | -------------: | -------------: |
| BioGRID | 305 | 208.3 | 143.1 | 159.0 |
| BioPlex | 189 | 138.6 | 71.4 | 111.0 |
| ComPPIHumanInt | 301 | 204.1 | 138.7 | 159.0 |
| ConsensusPathDB | 298 | 207.4 | 140.8 | 161.5 |
| FunCoup | 299 | 204.7 | 139.4 | 158.0 |
| HIPPIE | 306 | 208.1 | 142.9 | 159.5 |
| HuMAP | 279 | 194.3 | 126.7 | 155.0 |
| HuRI | 152 | 122.9 | 54.7 | 108.0 |
| HumanBaseTopGlobal | 287 | 219.7 | 145.7 | 173.0 |
| HumanNet | 302 | 204.2 | 140.3 | 158.5 |
| OmniPath | 298 | 199.6 | 136.0 | 153.5 |
| PCNet | 292 | 202.1 | 135.5 | 159.0 |
| ProteomeHD | 56 | 78.0 | 24.8 | 71.0 |
| SIGNOR | 219 | 147.3 | 81.9 | 124.0 |
| STRING | 296 | 208.0 | 140.6 | 162.0 |

#### GOBP

| Network | Num. tasks | Num. pos. avg. | Num. pos. std. | Num. pos. med. |
| :------ | ---------: | -------------: | -------------: | -------------: |
| BioGRID | 114 | 89.5 | 37.1 | 76.0 |
| BioPlex | 38 | 77.6 | 22.6 | 76.0 |
| ComPPIHumanInt | 104 | 91.8 | 37.0 | 77.5 |
| ConsensusPathDB | 112 | 90.1 | 37.0 | 76.5 |
| FunCoup | 114 | 87.8 | 36.7 | 74.0 |
| HIPPIE | 111 | 89.2 | 37.1 | 76.0 |
| HuMAP | 96 | 84.6 | 32.3 | 74.0 |
| HuRI | 27 | 69.9 | 16.0 | 65.0 |
| HumanBaseTopGlobal | 115 | 89.2 | 37.3 | 76.0 |
| HumanNet | 117 | 88.6 | 36.9 | 75.0 |
| OmniPath | 106 | 88.7 | 36.2 | 74.0 |
| PCNet | 105 | 89.0 | 36.0 | 77.0 |
| ProteomeHD | 5 | 80.4 | 22.6 | 70.0 |
| SIGNOR | 41 | 81.3 | 22.7 | 78.0 |
| STRING | 116 | 88.9 | 36.6 | 75.0 |
