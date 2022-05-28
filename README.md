# xtx
- Put train.csv and test.csv to directory data
- Prepare flatten features (can take long time, please let me know if it's a problem)
```bash
python -m xtx.scripts.extract_flatten_features
```
- Run
```bash
python -m xtx.scripts.train
```
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
- extract features:
    - common sense features
    - topk row aggregations from flatten features:
    Example: askRate0 = 1000, askRate1 = 1001, askSize0= 1, askSize1: 2
    askFlattenFeatures = [1000, 1001, 1001]
    - time features: different time aggregations of common sense features with small period
- 5Fold and 10Fold cross-validation with minibatches