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
    max_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    log_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=-float("inf"))
        block_max = tl.max(a, axis=0)
        max_acc_new = tl.where(max_acc > block_max, max_acc, block_max)

        raw_exp =  tl.math.exp(a - max_acc_new)

        log_acc_new = tl.math.exp(max_acc - max_acc_new) * log_acc + tl.sum(raw_exp, axis=-1)

        log_acc = log_acc_new
        max_acc = max_acc_new

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    o = tl.math.exp(x - max_acc) / log_acc
    tl.store(output_ptr + offset, o, mask=mask)

# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    softmax_kernel[grid](
        input_ptr, output_ptr, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
