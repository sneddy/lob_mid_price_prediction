# from collections import OrderedDict
# from functools import cached_property, partial
# from multiprocessing.pool import ThreadPool
# from typing import Dict, List

# from tqdm import tqdm


# class SparseBook:
#     def __init__(
#         self,
#         book_dict: List[Dict[float, int]],
#         ask: bool,
#         aligned: bool = False,
#         n_jobs: int = 1,
#     ):
#         self.book_dict = book_dict
#         self.ask = ask
#         self.aligned = aligned
#         self.n_jobs = n_jobs

#     def _cutout_record(self, record, n_operations: int, fillna: bool = False):
#         output_dict = OrderedDict()
#         n_rest = n_operations
#         for rate, size in record.items():
#             if size < n_rest:
#                 output_dict[rate] = size
#                 n_rest -= size
#             else:
#                 output_dict[rate] = n_rest
#                 n_rest = 0
#                 break
#         if fillna and n_rest > 0:
#             rate = rate + 1 if self.ask else rate - 1
#             output_dict[rate] = n_rest
#         return output_dict

#     def cutout(self, n_operations: int, fillna: bool = False):
#         if self.n_jobs == 1:
#             cutout_records = [
#                 self._cutout_record(record, n_operations, fillna)
#                 for record in self.book_dict
#             ]
#         else:
#             pool = ThreadPool(processes=self.n_jobs)
#             fn = partial(self._cutout_record, n_operations=n_operations, fillna=fillna)
#             cutout_records = pool.map(fn, [record for record in self.book_dict])
#         return SparseBook(cutout_records, self.ask, aligned=fillna, n_jobs=self.n_jobs)

#     def __len__(self):
#         return len(self.book_dict)

#     @cached_property
#     def mean(self):
#         outputs = []
#         for record in self.book_dict:
#             cumsum = 0
#             n = 0
#             for price, size in record.items():
#                 cum_sum += price * size
#                 n += size
#             outputs.append(cumsum / n)
#         return outputs

#     @cached_property
#     def row_len(self):
#         if self.aligned:
#             first_row_len = sum(self.book_dict[0].values())
#             return [first_row_len] * len(self)

#         outputs = []
#         for row in self.book_dict:
#             outputs.append(sum(row.values()))
#         return outputs

#     @cached_property
#     def flatten(self):
#         """
#         Dangerous: oo much memory to use
#         """
#         output = []
#         for record in self.book_dict:
#             flatten = []
#             for price, size in record.items():
#                 flatten.extend([price] * size)
#             output.append(flatten)
#         return output

#     def agg_cached(self, op_fn):
#         """
#         Dangerous: Too much memory to use
#         """
#         output = []
#         for record in self.flatten:
#             output.append(op_fn(record))
#         return output

#     def agg(self, op_fn):
#         output = []
#         for record in self.book_dict:
#             flatten = []
#             for price, size in record.items():
#                 flatten.extend([price] * size)
#             output.append(op_fn(flatten))
#         return output

#     def __repr__(self):
#         output = []
#         for record in self.book_dict[:20]:
#             row_output = []
#             for rate, size in record.items():
#                 if rate == int(rate):
#                     rate = int(rate)
#                 row_output.append(f"{rate}:{int(size)}")
#             output.append("; ".join(row_output))
#         return "\n".join(output)


# class SparseBookFactory:
#     def __init__(self, df, n_jobs=1):
#         self.df = df
#         self.n_jobs = n_jobs

#     @cached_property
#     def ask_book(self):
#         return SparseBook(
#             self.make_book(self.df.iloc[:, :30]), ask=True, n_jobs=self.n_jobs
#         )

#     @cached_property
#     def bid_book(self):
#         return SparseBook(
#             self.make_book(self.df.iloc[:, 30:60]), ask=False, n_jobs=self.n_jobs
#         )

#     def _process_row(self, row):
#         outputs_dict = OrderedDict()
#         for idx in range(15):
#             rate, size = row[idx], row[15 + idx]
#             if size > 0:
#                 outputs_dict[rate] = int(size)
#         return outputs_dict

#     def make_book(self, df):
#         if self.n_jobs == 1:
#             return [self._process_row(row) for idx, row in df.iterrows()]
#         pool = ThreadPool(processes=self.n_jobs)
#         return pool.map(self._process_row, [row for idx, row in df.iterrows()])
