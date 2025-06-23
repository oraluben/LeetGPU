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


size = 1024 * 1024 * 30


a = torch.randn(size, device='cuda')
b = torch.randn(size, device='cuda')
c = torch.zeros(size, device='cuda')
c2 = torch.zeros_like(c)


triton_add(a, b, c, size)
tile_add(a, b, c2, size)

assert torch.allclose(c, c2)


print(f'triton: {benchmark(lambda: triton_add(a, b, c, size), 10, nvtx='triton')}')
print(f'tilelang: {benchmark(lambda: tile_add(a, b, c, size), 10, nvtx='tilelang')}')