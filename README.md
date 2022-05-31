# Run xtx short pipeline
- install requirements:
```bash
pip install -r requirements.txt
```
- Put train.csv and test.csv to directory data
- Add correct values for paths for train and test to all configs in configs/fast_experiments:
    - exp_usecols_0.yaml
    - exp_usecols_1.yaml
    - exp_usecols_2.yaml
    - exp_usecols_3.yaml
    - exp_usecols_4.yaml

- extract topk features:
```bash
python -m xtx.scripts.extract_topk_features_pool configs/exp_usecols_0.yaml
```

- extract and cache features
```bash
python -m xtx.scripts.prepare_features configs/exp_usecols_0.yaml
```

- run training pipeline
```bash
python -m xtx.scripts.train configs/fast_experiments/exp_usecols_0.yaml
python -m xtx.scripts.train configs/fast_experiments/exp_usecols_1.yaml
python -m xtx.scripts.train configs/fast_experiments/exp_usecols_2.yaml
python -m xtx.scripts.train configs/fast_experiments/exp_usecols_3.yaml
python -m xtx.scripts.train configs/fast_experiments/exp_usecols_4.yaml
```

- run final ensembling
```bash
python -m xtx.scripts.ensemble_experiments
```

- grab prediction from predictions/averaged_predictions.npy

## Description

For this exercise we have a prediction problem.

The set-up is the following:

1 - For each point you have target variable "y", and 60 features (4 groups of 15 features representing the state of the order book at each point in time).

2 - Whenever there is a NaN in any of askRate,bidRate,askSize,bidSize, it means that the book is empty beyond that level (you can interpret it as the size being zero).

3 - Your forecast of y(k) can depend on askRate[:k+1,:], bidRate[:k+1,:], askSize[:k+1,:], bidSize[:k+1,:], i.e. all history including the current observation.

4 - The objective function is mean squared error, and you will be evaluated based on the out-of-sample performance of the model.

You can use the programming language/packages of your choice.

Ideally, your solution should have ‘train’ part that will use data.csv file for training and produce a model; and ‘predict’ part that will load data in the same format (but without ‘y’) and produce a prediction.

You'll have writing access to it so that once you have finished, you can upload your solution. There is no particular deadline, you can take your time doing this.

As for benchmark scores, on test dataset 0.17 correlation is very good, while 0.15 correlation is mediocre (can be achieved with reasonable linear model). (used correlation here just for optics, objective function is MSE)

## Solution Overview
- You can check notebooks/demo.ipynb

## Feature Selection
Selected columns can be reproduces by
```bash
python -m xtx.scripts.select_features
```
Trained large pool of reasonable features (About 230)
After that on every fold applied strategy:
- sample half of them randomly
- train and validate simple Ridge with relatively small alpha (~10).
- greedy insert and remove features. Aplly changes only if validation score on this fold increased
- after some kind convergence use this features to train stacking/ensembling on every fold.
- Builded features kindo diverse for every fold so can be good for ensembling.

## Metrics
Some kind of Kaggle Style improvement can be easily done by running more experiments and adding them to stacking:
- averaging predictions by other pseudo targets (different shifts, some smoothing)
- averaging predictions by different time folds
- adding another lgbm models to model_zoo

Non trivial improvements can be done by training some neural networks.
My recent experiments wasn't successful
- https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/blob/master/jupyter_pytorch/run_train_pytorch.ipynb


## Machine
- All calculations was performed on single macbook pro so maybe score will be higher after conducting more experiments
