#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
// A, B, and C are device pointers
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int tid = threadIdx.x;
    __shared__ float As[BM * BK], Bs[BK * BN];
    int cRow = blockIdx.y, cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    float pdc[TM * TN] = {};
    int Bcl = blockDim.x / BK;
    int ar = tid / BK, ac = tid % BK, br = tid / Bcl, bc = tid % Bcl, cr = tid / (BN / TN) * TM,
        cc = tid % (BN / TN) * TN;
    for (int t = 0; t < K; t += BK) {
        for (int s = 0; s < BM; s += Bcl) {
            if (t + ac < K && s + ar + cRow * BM < M) {
                As[ac + BK * (ar + s)] = A[ac + K * (ar + s)];
            } else
                As[ac + BK * (s + ar)] = 0.0f;
            if (s + bc + BN * cCol < N && t + br < K) {
                Bs[bc + s + BN * br] = B[bc + s + N * br];
            } else
                Bs[bc + s + BN * br] = 0.0f;
        }
        __syncthreads();

        A += BK;
        B += N * BK;
        float regM[TM], regN[TN];

        for (int ci = 0; ci < BK; ++ci) {
            for (int i = 0; i < TM; ++i) regM[i] = As[BK * (cr + i) + ci];

            for (int j = 0; j < TN; ++j) regN[j] = Bs[BN * ci + cc + j];

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    pdc[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            if (cRow * BM + cr + i < M && cCol * BN + cc + j < N) {
                int idx = N * (cr + i) + cc + j;
                float temp = alpha * pdc[j + i * TN];

                if (beta != 0.0f) temp += beta * __half2float(C[idx]);

                C[idx] = __float2half(temp);
            }
        }
    }
}
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 blocksPerGrid((N + BM - 1) / BM, (M + BN - 1) / BN), threadsPerBlock(BM * BN / (TM * TN));
    gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}