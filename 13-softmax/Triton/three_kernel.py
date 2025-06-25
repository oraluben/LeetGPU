import triton
import triton.language as tl

@triton.jit
def partial_max_value_kernel(X, partial_max, N, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    partial_max = partial_max.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=-float("inf"))
    local_max = tl.max(x, axis=0)
    tl.store(partial_max + pid, local_max)

@triton.jit
def partial_exp_sum_value_kernel(X, partial_sum, global_max, N, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    partial_sum = partial_sum.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask, other=-float("inf"))
    gmax = tl.load(global_max)
    local_sum = tl.sum(tl.exp(x - gmax), axis=0)
    tl.store(partial_sum + pid, local_sum)

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

@triton.jit
def get_max_value(partial_max, global_max, BLOCK_SIZE: tl.constexpr):
    partial_max = partial_max.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))
    offset =  tl.arange(0, BLOCK_SIZE)
    x = tl.load(partial_max + offset)
    local_max = tl.max(x, axis=0)
    tl.store(global_max, local_max)

@triton.jit
def get_sum_value(partial_sum, global_sum, BLOCK_SIZE: tl.constexpr):
    partial_sum = partial_sum.to(tl.pointer_type(tl.float32))
    global_sum = global_sum.to(tl.pointer_type(tl.float32))
    offset =  tl.arange(0, BLOCK_SIZE)
    x = tl.load(partial_sum + offset)
    local_sum = tl.sum(x, axis=0)
    tl.store(global_sum, local_sum)

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
    partial_max = cudaEmpty(BLOCK_SIZE)
    partial_max_value_kernel[grid](
        input_ptr, partial_max, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    global_max = cudaEmpty(1)
    get_max_value[1,](partial_max, global_max, BLOCK_SIZE=num_blocks)
    partial_sum = cudaEmpty(num_blocks)
    partial_exp_sum_value_kernel[grid](
        input_ptr, partial_sum, global_max, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    global_sum = cudaEmpty(1)
    get_sum_value[1,](partial_sum, global_sum, BLOCK_SIZE=num_blocks)
    normalize_kernel[grid](
        input_ptr, output_ptr, N,
        global_max, global_sum,
        BLOCK_SIZE=BLOCK_SIZE
    )
