import tilelang
import tilelang.language as T

import torch


@tilelang.jit
def add(a, b, c, M, block_M, dtype="float32"):
    a: T.Tensor((M,), dtype)
    b: T.Tensor((M,), dtype)
    c: T.Tensor((M,), dtype)

    num_per_thread = 8

    with T.Kernel(T.ceildiv(M, block_M * num_per_thread), threads=128) as bx:
        for local_x, i in T.Parallel(block_M, num_per_thread):
            x = (bx * block_M + local_x) * num_per_thread
            c[x + i] = a[x + i] + b[x + i]
            pass


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 128

    add(a, b, c, N, BLOCK_SIZE)
