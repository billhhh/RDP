# RDP

Codes for IJCAI2020 paper "Unsupervised Representation Learning by Predicting Random Distances” https://arxiv.org/abs/1912.12186


## Installation

The repo is tested on Ubuntu 16.04, Python 3.5.2, PyTorch 1.1.0 and Sklearn 0.21.1.


## Anomaly Detection

### Data Preparation

Some of example datasets are put in ./data folder due to the large file size limitation. You may downloaded them from the urls listed in the paper appendix.

### Train

If you are under Dev mode (tweak it in train.py), just run

```
python train.py
```

If you are under Server mode, the following scripts can be used to help you run experiments in batch

```
python train.py [data/csv_file] [save_path] > [output_log] 2>&1 &
```

e.g.

```
python train.py data/apascal.csv save_model/apascal/ > logs/apascal.log 2>&1 &
...
```

### Test

If you are under Dev mode, just run

```
python test.py
```

If you are under Server mode, the following scripts can be used to help you run experiments in batch

```
python test.py [data/csv_file] [load_path] [tree_depth] [testing_method] > [output_log] 2>&1 &
```

e.g.

```
python test.py data/apascal.csv save_model/apascal/ 8 1 > logs/apascal_l8_test.log 2>&1 &
...
```
