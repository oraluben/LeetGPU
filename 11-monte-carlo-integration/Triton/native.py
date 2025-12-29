# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def monte_carlo_kernel(y_ptr, result_ptr, N, a, b, BLOCK_SIZE: tl.constexpr):
    y_ptr = y_ptr.to(tl.pointer_type(tl.float32))
    result_ptr = result_ptr.to(tl.pointer_type(tl.float32))
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        y = tl.load(y_ptr + offsets, mask=offsets < N, other=0)
        _sum += y
    sum = tl.sum(_sum, axis=0)
    tl.store(result_ptr, (b - a) * sum / N)

# y_samples_ptr, result_ptr are raw device pointers
def solve(y_samples_ptr: int, result_ptr: int, a: float, b: float, n_samples: int):
    BLOCK_SIZE = 32768
    grid = (triton.cdiv(n_samples, BLOCK_SIZE),)

    monte_carlo_kernel[grid](
        y_samples_ptr,
        result_ptr,
        n_samples,
        a, b,
        BLOCK_SIZE=BLOCK_SIZE
    )