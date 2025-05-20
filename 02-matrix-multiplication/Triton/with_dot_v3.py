import triton
import triton.language as tl

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_k = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_k

@triton.jit
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_an,  #
                             stride_bn, stride_bk,  #
                             stride_cm, stride_ck,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_k

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_n_for_mask = tl.arange(0, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_k = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_k = pid_k * BLOCK_SIZE_K
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bk = start_k + tl.arange(0, BLOCK_SIZE_K)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bk = tl.where(offs_bk < K, offs_bk, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bk = tl.max_contiguous(tl.multiple_of(offs_bk, BLOCK_SIZE_K), BLOCK_SIZE_K)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for ni in range(n_tiles):
            offs_k = ni * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_an)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bn + offs_bk[None, :] * stride_bk)

            a = tl.load(a_ptrs, mask=offs_n_for_mask[None, :] < N - ni * BLOCK_SIZE_N, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n_for_mask[:, None] < N - ni * BLOCK_SIZE_N, other=0.0)
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")

        tile_id_c += NUM_SMS
        pid_m, pid_k = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_ck * offs_ck[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
        c = accumulator.to(tl.float32)
        tl.store(c_ptrs, c, mask=c_mask)

# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1
    # SM count for Tesla T4 (avoid torch API)
    # you can get the value with: torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_SMS = 40
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"])), )
    kernel = matmul_kernel_persistent[grid](
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,  #
    )
