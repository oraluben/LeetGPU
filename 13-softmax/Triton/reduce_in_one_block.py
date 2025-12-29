# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    _max = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=-float("inf"))
        _max = tl.maximum(a, _max)
    max = tl.max(_max, axis=0)
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=-float("inf"))
        _sum += tl.exp(a - max)
    sum = tl.sum(_sum, axis=0)
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    y = tl.exp(x - max) / sum
    tl.store(output_ptr + offset, y, mask=mask)

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 32768
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    softmax_kernel[grid](
        input_ptr, output_ptr, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
