import time

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from xtx.flatten_tools import ask_flatten, flatten
from xtx.sparse_book import SparseBookFactory

pandarallel.initialize(progress_bar=True)



def main():
    data = pd.read_pickle('data/data.pkl')
    target_col = 'y'
    
    # df = data.head(1000000)
    # t0 = time.time()
    # sparse_book_factory = SparseBookFactory(df, n_jobs=8)
    # ask_book = sparse_book_factory.ask_book
    # duration = time.time() - t0
    # print(f'Init time: {duration:.4f}')

    # t0 = time.time()
    # cutout_book = ask_book.cutout(1000, fillna=True)
    # duration = time.time() - t0
    # print(f'Cutout time: {duration:.4f}')
    
    # t0 = time.time()
    # cutout_book = cutout_book.agg(np.mean)
    # duration = time.time() - t0
    # print(f'Agg time: {duration:.4f}')

    t0 = time.time()
    flatten_df = data.head(1000000).parallel_apply(lambda x: ask_flatten(x, n=1000), axis=1) 
    duration = time.time() - t0
    print(f'Flatten time: {duration:.4f}')

    t0 = time.time()
    x = flatten_df.parallel_apply(np.mean)
    duration = time.time() - t0
    print(f'Flatten Operation time: {duration:.4f}')

if __name__ == '__main__':
    main()
