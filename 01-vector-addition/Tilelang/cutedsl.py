import tilelang
import tilelang.language as T

import torch


@tilelang.jit(target='cutedsl')
def add(M, block_M, dtype="float32"):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, ), dtype),
        B: T.Tensor((M, ), dtype),
        C: T.Tensor((M, ), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            start_x = bx * block_M
            for local_x in T.Parallel(block_M):
                x = start_x + local_x
                C[x] = A[x] + B[x]

    return add_kernel


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 128
    jit_kernel = add(N, BLOCK_SIZE)

    jit_kernel(a, b, c)
