// refer to https://zhuanlan.zhihu.com/p/692010210 for more details.

#include "solve.h"
#include <cuda_runtime.h>

template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v3(const float* idata, float* odata, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    if (x < N) {
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            if (y + y_off < M) {
                sdata[ty + y_off][tx] = idata[(y + y_off) * N + x];
            }
        }
    }
    __syncthreads();
    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (x < M) {
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            if (y + y_off < N) {
                odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
            }
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    constexpr int BLOCK_SZ = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_THREAD);
    dim3 grid((cols+ BLOCK_SZ-1)/BLOCK_SZ, (rows+BLOCK_SZ-1)/BLOCK_SZ);
    mat_transpose_kernel_v3<BLOCK_SZ, NUM_PER_THREAD><<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}