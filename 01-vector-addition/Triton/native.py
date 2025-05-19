# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    # Multiple "programs" are processing different data chunks. Here we determine which one we are
    pid = tl.program_id(axis=0)  # We launch a 1D grid, so the axis is 0.
    # This program will handle input starting from a certain offset.
    # For example, if the vector length is 256 and block size is 64,
    # programs will access elements [0:64), [64:128), [128:192), [192:256) respectively.
    block_start = pid * BLOCK_SIZE
    # Note: `offsets` is a list of pointers.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    # Load a and b from DRAM; the mask ensures we avoid reading beyond the input size
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    # Write a + b back to DRAM
    tl.store(c_ptr + offsets, c, mask=mask)

# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, N: int):
    # 128/256 is the optimal parameter
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE)
