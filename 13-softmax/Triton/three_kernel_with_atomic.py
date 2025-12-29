import triton
import triton.language as tl

@triton.jit
def partial_max_value_kernel(X, global_max, N, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))
    tl.store(global_max, -float("inf"))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=-float("inf"))
    local_max = tl.max(x, axis=0)
    tl.atomic_max(global_max, local_max)


@triton.jit
def partial_exp_sum_value_kernel(X, global_sum, global_max, N, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    global_sum = global_sum.to(tl.pointer_type(tl.float32))
    tl.store(global_sum, 0)
    global_max = global_max.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=-float("inf"))
    gmax = tl.load(global_max)
    local_sum = tl.sum(tl.exp(x - gmax), axis=0)
    tl.atomic_add(global_sum, local_sum)


@triton.jit
def normalize_kernel(X, Y, N, global_max, global_sum, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    Y = Y.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))
    global_sum = global_sum.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    gmax = tl.load(global_max)
    gsum = tl.load(global_sum)
    y = tl.exp(x - gmax) / gsum
    tl.store(Y + offset, y, mask=mask)


def cudaEmpty(num_elements:int):
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(ptr), num_elements*4)
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed, code {err}")
    return ptr.value


# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 32768
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)
    global_max = cudaEmpty(1)
    partial_max_value_kernel[grid](
        input_ptr, global_max, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    global_sum = cudaEmpty(1)
    partial_exp_sum_value_kernel[grid](
        input_ptr, global_sum, global_max, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    normalize_kernel[grid](
        input_ptr, output_ptr, N,
        global_max, global_sum,
        BLOCK_SIZE=BLOCK_SIZE
    )
