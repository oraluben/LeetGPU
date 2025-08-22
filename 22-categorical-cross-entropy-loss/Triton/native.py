# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl
import numpy as np
import ctypes

@triton.jit
def cross_entropy_kernel(
    logits_ptr,         # [N, C]
    labels_ptr,         # [N]
    loss_ptr,           # [N]
    N, C,
    BLOCK_SIZE: tl.constexpr,
):
    logits_ptr = logits_ptr.to(tl.pointer_type(tl.float32))
    labels_ptr = labels_ptr.to(tl.pointer_type(tl.int32))
    loss_ptr = loss_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    offs = pid * C + tl.arange(0, BLOCK_SIZE)

    class_ids = tl.arange(0, BLOCK_SIZE)

    logits = tl.load(logits_ptr + offs, mask=(class_ids < C), other=-float("inf"))

    max_logits = tl.max(logits, axis=0)

    logits_norm = logits - max_logits
    exp_logits = tl.exp(logits_norm)

    sum_exp = tl.sum(exp_logits, axis=0)

    log_sum_exp = tl.log(sum_exp) + max_logits

    labels = tl.load(labels_ptr + pid)
    true_class_offsets = pid * C  + labels
    z_y = tl.load(logits_ptr + true_class_offsets)

    loss = log_sum_exp - z_y
    tl.store(loss_ptr + pid, loss)

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

# logits_ptr, true_labels_ptr, loss_ptr are raw device pointers
def solve(logits_ptr: int, true_labels_ptr: int, loss_ptr: int, N: int, C: int):
    loss = cudaEmpty(N)
    cross_entropy_kernel[(N,)](
        logits_ptr, true_labels_ptr, loss,
        N, C,
        BLOCK_SIZE=1024,
    )
    dtype = np.float32
    host_arr = np.empty(N, dtype=dtype)
    libcudart = ctypes.CDLL('libcudart.so')
    cudaMemcpy = libcudart.cudaMemcpy
    cudaMemcpy.restype = ctypes.c_int
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpy(
        host_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(loss),
        host_arr.nbytes,
        cudaMemcpyDeviceToHost
    )
    mean_arr = np.array([np.mean(host_arr)], dtype=dtype)
    cudaMemcpy(
        ctypes.c_void_p(loss_ptr),
        mean_arr.ctypes.data_as(ctypes.c_void_p),
        4,
        cudaMemcpyHostToDevice
    )