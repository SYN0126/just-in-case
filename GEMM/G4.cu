
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8

// Kernel 4: 1D Register Tiling
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int tid = threadIdx.x;

    __shared__ float As[BM * BK], Bs[BK * BN];

    int cRow = blockIdx.y, cCol = blockIdx.x;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float pdc[TM] = {};

    int ar = tid / BK, ac = tid % BK;
    int br = tid / BN, bc = tid % BN;
    int cr = tid / BN * TM, cc = tid % BN;

    for (int t = 0; t < K; t += BK) {
        // Global Memory -> Shared Memory
        if (t + ac < K && ar + cRow * BM < M) {
            As[ac + BK * ar] = A[ac + K * ar];
        } else {
            As[ac + BK * ar] = 0.0f;
        }

        if (bc + BN * cCol < N && t + br < K) {
            Bs[bc + BN * br] = B[bc + N * br];
        } else {
            Bs[bc + BN * br] = 0.0f;
        }

        __syncthreads();

        // 1D Register Tiling
        for (int ci = 0; ci < BK; ++ci) {
            float btemp = Bs[BN * ci + cc];

            for (int resid = 0; resid < TM; ++resid) {
                pdc[resid] += As[BK * (cr + resid) + ci] * btemp;
            }
        }

        __syncthreads();

        // Advance pointers
        if (t + BK < K) {
            A += BK;
            B += N * BK;
        }
    }

    // Write back C
    for (int i = 0; i < TM; ++i) {
        if (cRow * BM + cr + i < M && cCol * BN + cc < N) {
            int idx = N * (cr + i) + cc;
            float temp = alpha * pdc[i];

            if (beta != 0.0f) temp += beta * __half2float(C[idx]);

            C[idx] = __float2half(temp);
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 threadsPerBlock(BM * BN / TM);

    gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);

    cudaDeviceSynchronize();
}
