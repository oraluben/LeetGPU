import triton
import triton.language as tl

@triton.jit
def qkt_kernel(Q_ptr, K_ptr, Scores_ptr, scale,
              M, N, D, h,
              stride_qm, stride_qd,
              stride_kn, stride_kd,
              stride_sm, stride_sn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    Q_ptr = Q_ptr.to(tl.pointer_type(tl.float32))
    K_ptr = K_ptr.to(tl.pointer_type(tl.float32))
    Scores_ptr = Scores_ptr.to(tl.pointer_type(tl.float32))
    d_model = D * h
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    pid_h = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    Q_block_ptrs = Q_ptr + pid_h * D + offs_m[:, None] * d_model + offs_d[None, :]
    K_block_ptrs = K_ptr + pid_h * D + offs_d[:, None] + offs_n[None, :] * d_model

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for d_off in range(0, D, BLOCK_D):
        Q_block = tl.load(Q_block_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] + d_off < D), other=0.0)
        K_block = tl.load(K_block_ptrs, mask=(offs_d[:, None] + d_off < D) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(Q_block, K_block, input_precision="ieee")

        Q_block_ptrs += BLOCK_D * stride_qd
        K_block_ptrs += BLOCK_D * stride_kd

    Scores_ptrs = Scores_ptr + pid_h * M * N + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    tl.store(Scores_ptrs, acc *  scale, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, h,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    pid_m = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    pid_h = tl.program_id(axis=0)
    d_model = K * h

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + pid_h * M * N + offs_m[:, None] * N + offs_n[None, :]
    b_ptrs = b_ptr + pid_h * K + offs_n[:, None] * d_model + offs_k[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a = tl.load(a_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N * d_model

    c_ptrs = c_ptr + pid_h * K + offs_m[:, None] * d_model + offs_k[None, :]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, N, h,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    rows = 131072 // BLOCK_SIZE
    pid = tl.program_id(0)
    start = pid * rows
    end = tl.minimum((start + rows), h*N)
    cols =  tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    for row in range(start, end):
        input_ptrs = input_ptr + row * N
        output_ptrs = output_ptr + row * N
        a = tl.load(input_ptrs + cols, mask=mask, other=-float("inf"))
        max = tl.max(a, axis=0)
        sum = tl.sum(tl.exp(a-max), axis=0)
        y = tl.exp(a - max) / sum
        tl.store(output_ptrs + cols, y, mask=mask)


def cudaEmpty(num_elements: int):
    # return torch.empty(num_elements, device="cuda", dtype=torch.float32)
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(ptr), num_elements*4)
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed, code {err}")
    return ptr.value



# Q, K, V, output are raw device pointers
def solve(Q_ptr: int, K_ptr: int, V_ptr: int, output_ptr: int, N: int, d_model: int, h: int):
    M = N
    d = d_model // h
    scores = cudaEmpty(h*M*N)
    scale = 1.0 / (d ** 0.5)
    grid = lambda META: (h, triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    qkt_kernel[grid](
        Q_ptr, K_ptr, scores, scale,
        M, N, d, h,
        d, 1,  # Q strides
        d, 1,  # K strides
        N, 1,  # Output strides
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_D=64,
    )
    softmax = cudaEmpty(h*M*N)
    softmax_block_size = triton.next_power_of_2(N)
    softmax_grid = (triton.cdiv(h * N, 131072 // softmax_block_size),)
    softmax_kernel[softmax_grid](
        scores, softmax, N, h,
        BLOCK_SIZE = softmax_block_size,
    )
    matmul_grid = lambda META: (h, triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(d, META['BLOCK_SIZE_K']))
    matrix_multiplication_kernel[matmul_grid](
        softmax, V_ptr, output_ptr,
        M, N, d, h,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64
    )
