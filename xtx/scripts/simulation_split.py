import os

import pandas as pd


def make_split(data, test_ratio: float, train_test_gap: int = 0):
    test_size = int(data.shape[0] * test_ratio)
    train_size = data.shape[0] - test_size - train_test_gap
    train_data = data.iloc[:train_size, :]
    test_data = data.iloc[-test_size:, :]
    return train_data, test_data


def main():
    data = pd.read_csv("data/xtx_data.csv")
    test_ratio = 0.2
    train_test_gap = 2000
    output_dir = "data/simulation"
    os.makedirs(output_dir, exist_ok=True)
    train, test = make_split(data, test_ratio, train_test_gap=train_test_gap)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
