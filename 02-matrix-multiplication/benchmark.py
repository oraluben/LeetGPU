import torch

from Triton.native import solve as triton_gemm
from Tilelang.native import solve as tile_gemm


def benchmark(f, n=None, nvtx=None, *args, **kwargs):
    # trigger jit
    f(*args, **kwargs)

    if n is None:
        assert False, 'TODO: estimate time'

    if nvtx:
        torch.cuda.nvtx.range_push(nvtx)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(n):
        f(*args, **kwargs)

    end.record()
    end.synchronize()

    if nvtx:
        torch.cuda.nvtx.range_pop()

    return start.elapsed_time(end) / 1000


size = 1024

a = torch.randn(size, size, device='cuda', dtype=torch.float16)
b = torch.randn(size, size, device='cuda', dtype=torch.float16)
c = torch.zeros(size, size, device='cuda', dtype=torch.float16)

# triton_gemm(a, b, c, size, size)
tile_gemm(a, b, c, size, size, size)

assert torch.allclose(c, a @ b)

y1, y2, y3 = [], [], []

y1.append(benchmark(lambda: triton_gemm(a, b, c, size, size, size), 100, nvtx='triton'))
y2.append(benchmark(lambda: tile_gemm(a, b, c, size, size, size), 100, nvtx='tilelang'))
y3.append(benchmark(lambda: torch.matmul(a, b, out=c), 100, nvtx='torch'))

print(y1, y2, y3)


# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# df1 = pd.DataFrame({'x': sizes, 'y': y1})
# df2 = pd.DataFrame({'x': sizes, 'y': y2})
# df3 = pd.DataFrame({'x': sizes, 'y': y3})

# plt.figure(figsize=(10, 6))
# ax = plt.gca()

# sns.regplot(x='x', y='y', data=df1, ax=ax, label='Triton', scatter_kws={'s': 50})
# sns.regplot(x='x', y='y', data=df2, ax=ax, label='Tilelang', scatter_kws={'s': 50})
# sns.regplot(x='x', y='y', data=df3, ax=ax, label='Torch', scatter_kws={'s': 50})

# plt.title('Vectorized add')
# plt.xlabel('Size')
# plt.ylabel('Time')

# plt.legend()

# plt.savefig('01-result.png')
