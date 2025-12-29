import cutlass.cute as cute


@cute.kernel
def vector_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        C[thread_idx] = A[thread_idx] + B[thread_idx]


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    num_threads_per_block = 256
    grid_dim = cute.ceil_div(cute.Int32(N), num_threads_per_block), 1, 1
    vector_add_kernel(A, B, C, N).launch(
        grid=grid_dim, block=(num_threads_per_block, 1, 1)
    )


import torch


def compiled_solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int):
    assert a.dtype == torch.float32
    a_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    b_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    c_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
    kernel = cute.compile(
        vector_add_kernel, a_fake, b_fake, c_fake, n, options="--enable-tvm-ffi"
    )
    kernel(a, b, c, n)
