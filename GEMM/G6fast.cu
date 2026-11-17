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
    __shared__ __align(16) __ float As[BM * BK], Bs[BK * BN];
    int cRow = blockIdx.y, cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    float pdc[TM * TN] = {};
    // int Bcl = blockDim.x / BK;
    int ar = tid / (BK / 4), ac = tid % (BK / 4) * 4, br = tid / (BN / 4), bc = tid % (BN / 4) * 4,
        cr = tid / (BN / TN) * TM, cc = tid % (BN / TN) * TN;
    for (int t = 0; t < K; t += BK) {
        const half2* ptr = reinterpret_cast<const half2*>(&A[ac + K * (ar)]);
        for (int i = 0; i < 4; ++i) {
            if (t + ac + i < K && ar + cRow * BM < M) {
                float v0 = i % 2 == 0 ? ptr[i / 2].x : ptr[i / 2].y;
                As[(i + ac) * BM + (ar)] = v0;
            } else
                As[(i + ac) * BM + (ar)] = 0.0f;
        }
        const half2* ptr1 = reinterpret_cast<const half2*>(&B[bc + N * br]);
        for (int i = 0; i < 4; ++i) {
            if (cCol * BN + bc + i < N && t + br < K) {
                float v0 = i % 2 == 0 ? ptr1[i / 2].x : ptr1[i / 2].y;
                Bs[i + bc + BN * (br)] = v0;
            } else
                Bs[i + bc + BN * (br)] = 0.0f;
        }
        __syncthreads();

        A += BK;
        B += N * BK;
        float regM[TM], regN[TN];

        for (int ci = 0; ci < BK; ci++) {
            for (int i = 0; i < TM; i += 4) {
                float4 v = reinterpret_cast<const float4*>(&As[cr + i + BM * ci])[0];

                regM[i] = v.x;
                regM[i + 1] = v.y;
                regM[i + 2] = v.z;
                regM[i + 3] = v.w;
            }

            for (int j = 0; j < TN; j += 4) {
                float4 v = reinterpret_cast<const float4*>(&Bs[BN * ci + cc + j])[0];

                regN[j] = v.x;
                regN[j + 1] = v.y;
                regN[j + 2] = v.z;
                regN[j + 3] = v.w;
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    pdc[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j += 2) {
            if (cRow * BM + cr + i < M && cCol * BN + cc + j < N) {
                int idx = N * (cr + i) + cc + j;
                float temp1 = alpha * pdc[j + i * TN];
                float temp2 = alpha * pdc[j + 1 + i * TN];

                half2* ptr = reinterpret_cast<half2*>(&C[idx]);

                if (beta != 0.0f) {
                    float2 old = __half22float2(ptr[0]);

                    temp1 += beta * old.x;
                    temp2 += beta * old.y;
                }

                ptr[0] = __floats2half2_rn(temp1, temp2);
            }
        }
    }
}
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 blocksPerGrid((N + BM - 1) / BM, (M + BN - 1) / BN), threadsPerBlock(BM * BN / (TM * TN));
    gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}