import tilelang
import tilelang.language as T

import torch

from functools import lru_cache


@lru_cache
def add(M, block_M, dtype="float32", accum_dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((M, ), dtype),
        B: T.Tensor((M, ), dtype),
        C: T.Tensor((M, ), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            start_x = bx * block_M
            for local_x in T.Parallel(block_M):
                x = start_x + local_x
                C[x] = A[x] + B[x]

    return main


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 128
    func = add(N, BLOCK_SIZE)
    jit_kernel = tilelang.compile(func, target="cuda")

    jit_kernel(a, b, c)
