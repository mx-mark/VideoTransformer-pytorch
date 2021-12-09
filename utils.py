import time
from pytorch_lightning.utilities.distributed import rank_zero_only

@rank_zero_only
def print_on_rank_zero(content):
    print(content)
    

def timeit_wrapper(func, *args, **kwargs):
    start = time.perf_counter()
    func_return_val = func(*args, **kwargs)
    end = time.perf_counter()
    return func_return_val, float(f'{end - start:.4f}')