import tilelang
import tilelang.language as T

import torch


@tilelang.jit
def add(M, block_M, dtype="float32"):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, ), dtype),
        B: T.Tensor((M, ), dtype),
        C: T.Tensor((M, ), dtype),
    ):
        num_per_thread = 8
        with T.Kernel(T.ceildiv(M, block_M * num_per_thread), threads=128) as bx:
            for local_x, i in T.Parallel(block_M, num_per_thread):
                x = (bx * block_M + local_x) * num_per_thread
                C[x + i] = A[x + i] + B[x + i]

    return add_kernel


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 128
    jit_kernel = add(N, BLOCK_SIZE)

    jit_kernel(a, b, c)
