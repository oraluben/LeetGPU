import torch

from Triton.native import solve as triton_add
from Tilelang.native import solve as tile_add


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


sizes = [128 * 2 ** p for p in range(0, 20)]
sizes = [127, 129] + sizes
y1 = []
y2 = []
y3 = []

for size in sizes:
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    c = torch.zeros(size, device='cuda')

    triton_add(a, b, c, size)

    assert torch.allclose(c, a + b)

    y1.append(benchmark(lambda: triton_add(a, b, c, size), 100, nvtx='triton'))
    y2.append(benchmark(lambda: tile_add(a, b, c, size), 100, nvtx='tilelang'))
    y3.append(benchmark(lambda: torch.add(a, b, out=c), 100, nvtx='torch'))


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.DataFrame({'x': sizes, 'y': y1})
df2 = pd.DataFrame({'x': sizes, 'y': y2})
df3 = pd.DataFrame({'x': sizes, 'y': y3})

plt.figure(figsize=(10, 6))
ax = plt.gca()

sns.regplot(x='x', y='y', data=df1, ax=ax, label='Triton', scatter_kws={'s': 50})
sns.regplot(x='x', y='y', data=df2, ax=ax, label='Tilelang', scatter_kws={'s': 50})
sns.regplot(x='x', y='y', data=df3, ax=ax, label='Torch', scatter_kws={'s': 50})

plt.title('Vectorized add')
plt.xlabel('Size')
plt.ylabel('Time')

plt.legend()

plt.savefig('01-result.png')
