import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_ir, stride_ic,
    stride_or, stride_oc,
    BLOCK_SIZE : tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    # 1. determine the input tile coordinates this thread block is responsible for
    pid_m = tl.program_id(0)  # block index in M dimension
    pid_n = tl.program_id(1)  # block index in N dimension

    # 2. compute element-wise offsets within the tile
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. define global memory pointers for input tile (row-major)
    input_ptrs = input_ptr + offs_m[:, None] * stride_ir + offs_n[None, :] * stride_ic

    # 4. load input tile from global memory with boundary check
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    block = tl.load(input_ptrs, mask=mask, other=0)

    # 5. transpose the tile (swap rows and columns)
    transposed_block = tl.trans(block)  # Triton built-in transpose function

    # 6. compute global memory pointers for output tile (column-major)
    output_ptrs = output_ptr + offs_n[:, None] * M + offs_m[None, :]  # M is row stride after transpose

    # 7. store the transposed tile to global memory
    tl.store(output_ptrs, transposed_block, mask=mask.T)  # transpose mask as well


# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = lambda META: (triton.cdiv(rows, META['BLOCK_SIZE']), triton.cdiv(cols, META['BLOCK_SIZE']))
    matrix_transpose_kernel[grid](
        input_ptr, output_ptr,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE=32
    )
