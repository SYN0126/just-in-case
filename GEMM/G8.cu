#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define PAD 5
// A, B, and C are device pointers
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int tid = threadIdx.x;
    int BNP = BN + PAD;
    __shared__ __align__(16) float As[BM * BK], Bs[BK * (BN + PAD)];
    int cRow = blockIdx.y, cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    float pdc[TM * TN] = {};
    int ar = tid / (BK / 4), ac = tid % (BK / 4) * 4, br = tid / (BN / 4), bc = tid % (BN / 4) * 4,
        cr = tid / (BN / TN) * TM, cc = tid % (BN / TN) * TN;
    for (int t = 0; t < K; t += BK) {
        if (ar + cRow * BM < M && t + ac + 3 < K && K % 2 == 0) {
            const half2* ptr = reinterpret_cast<const half2*>(&A[ac + K * ar]);

            half2 h0 = ptr[0];
            half2 h1 = ptr[1];

            As[(ac + 0) * BM + ar] = h0.x;
            As[(ac + 1) * BM + ar] = h0.y;
            As[(ac + 2) * BM + ar] = h1.x;
            As[(ac + 3) * BM + ar] = h1.y;

        } else {
            for (int i = 0; i < 4; ++i) {
                if (t + ac + i < K && ar + cRow * BM < M)
                    As[(ac + i) * BM + ar] = A[ac + i + K * ar];
                else
                    As[(ac + i) * BM + ar] = 0.0f;
            }
        }
        if (BN * cCol + 3 + bc < N && t + br < K && N % 2 == 0) {
            const half2* ptr1 = reinterpret_cast<const half2*>(&B[bc + N * br]);
            half2 h0 = ptr1[0];
            half2 h1 = ptr1[1];
            Bs[bc + 0 + BNP * br] = h0.x;
            Bs[bc + 1 + BNP * br] = h0.y;
            Bs[bc + 2 + BNP * br] = h1.x;
            Bs[bc + 3 + BNP * br] = h1.y;
        } else {
            for (int i = 0; i < 4; ++i) {
                if (cCol * BN + bc + i < N && t + br < K)
                    Bs[bc + i + BNP * br] = B[bc + i + N * br];
                else
                    Bs[bc + i + BNP * br] = 0.0f;
            }
        }
        __syncthreads();

        if (t + BK < K) {
            A += BK;
            B += N * BK;
        }
        float regM[TM], regN[TN];

        for (int ci = 0; ci < BK; ci++) {
            for (int i = 0; i < TM; i += 4) {
                float4 v = reinterpret_cast<const float4*>(&As[cr + i + BM * ci])[0];

                regM[i] = v.x;
                regM[i + 1] = v.y;
                regM[i + 2] = v.z;
                regM[i + 3] = v.w;
            }

            // for (int j = 0; j < TN; j += 4) {
            //     float4 v = reinterpret_cast<const float4*>(&Bs[BNP * ci + cc + j])[0];

            //     regN[j] = v.x;
            //     regN[j + 1] = v.y;
            //     regN[j + 2] = v.z;
            //     regN[j + 3] = v.w;
            // }
            for (int j = 0; j < TN; ++j) {
                regN[j] = Bs[BNP * ci + cc + j];
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
                if (cCol * BN + cc + j + 1 < N && N % 2 == 0) {
                    float temp1 = alpha * pdc[j + i * TN];
                    float temp2 = alpha * pdc[j + 1 + i * TN];

                    half2* ptr = reinterpret_cast<half2*>(&C[idx]);

                    if (beta != 0.0f) {
                        float2 old = __half22float2(ptr[0]);

                        temp1 += beta * old.x;
                        temp2 += beta * old.y;
                    }

                    ptr[0] = __floats2half2_rn(temp1, temp2);
                } else {
                    float temp = alpha * pdc[j + i * TN];
                    if (beta != 0.0f) {
                        temp += beta * __half2float(C[idx]);
                    }

                    C[idx] = __float2half(temp);
                    if (cCol * BN + cc + j + 1 < N) {
                        float temp1 = alpha * pdc[j + 1 + i * TN];

                        if (beta != 0.0f) temp1 += beta * __half2float(C[idx + 1]);

                        C[idx + 1] = __float2half(temp1);
                    }
                }
            }
        }
    }
}
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM), threadsPerBlock(BM * BN / (TM * TN));
    gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}