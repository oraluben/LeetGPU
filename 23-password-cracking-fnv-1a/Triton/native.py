# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(hash_val, byte):
    FNV_PRIME: tl.uint64 = 16777619
    return ((hash_val ^ byte) * FNV_PRIME) & 0xFFFFFFFF

@triton.jit
def password_cracker_kernel(target_hash1, target_hash2, R, output_password_ptr, length):
    output_password_ptr = output_password_ptr.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(0)
    idx = pid
    OFFSET_BASIS : tl.uint32 = 2166136261
    h : tl.uint32 = OFFSET_BASIS
    for i in range(length):
        h = fnv1a_hash(h, (idx % 26) + 97)
        idx = idx // 26
    for _ in range(R - 1):
        tmp = h
        h = OFFSET_BASIS
        for i in range(0, 4):
            h = fnv1a_hash(h, tmp % 256)
            tmp = tmp // 256
    # Compare
    idx = pid
    if h == target_hash1*65536+target_hash2:
        for i in range(length):
            tl.store(output_password_ptr+i, (idx % 26) + 97)
            idx = idx // 26

# target_hash is a value, output_password_ptr is a raw device pointer
def solve(target_hash: int, password_length: int, R: int, output_password_ptr: int):
    target_hash1, target_hash2 = target_hash // 65536, target_hash%65536
    if password_length > 6:
        return
    total: int = 26 ** password_length
    password_cracker_kernel[(total,)](
        target_hash1, target_hash2, R, output_password_ptr, password_length,
        num_warps=1,
    )