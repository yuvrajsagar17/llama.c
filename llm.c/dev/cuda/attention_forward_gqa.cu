#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CUDA & cuDNN setup
static bool first_run_validation = true; // always run e.g. permute on 1st run

#ifdef ENABLE_CUDNN
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#if CUBLAS_LOWP == CUDA_R_16BF
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#else
#define CUDNN_16BIT fe::DataType_t::HALF
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void *cudnn_workspace = NULL;

#define checkCudaErr(err) assert((int)err == 0);
#define checkCudnnErr(err) assert((int)err == 0);
#endif // ENABLE_CUDNN
// ----------------------------------------------------------------------------

void attention_forward_gqa(float *out, float *qkvr, float *att, float *inp,
                           float *freq_cos, float *freq_sin,
                           int B, int T, int C, int NH, int num_kv_heads)
{
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size
    int queries_per_kv = NH / num_kv_heads;

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    // size_t ksize = sizeof(k) / sizeof(k[0]);
    // size_t vsize = sizeof(v) / sizeof(v[0]);
    // printf("ksize-Vsize: %ld, %ld\n%d, %d, %d\n", ksize, vsize, HS, kv_HS, queries_per_kv);
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);

    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    apply_rope_forward(q, k, freq_cos, freq_sin, B, T, NH, (C / NH));

    // Repeat interleave for GQA
    if (num_kv_heads != NH)
    {
        float *new_k, *new_v;
        cudaMalloc((void **)&new_k, B * NH * T * HS * sizeof(float));
        cudaMalloc((void **)&new_v, B * NH * T * HS * sizeof(float));

        int repeat_interleave_threads = B * num_kv_heads * queries_per_kv * T * HS;
        repeat_interleave_forward_kernel<<<num_blocks, block_size>>>(new_k, k, B, num_kv_heads, T, HS, queries_per_kv);
        repeat_interleave_forward_kernel<<<num_blocks, block_size>>>(new_v, v, B, num_kv_heads, T, HS, queries_per_kv);
        cudaCheck(cudaGetLastError());

        // Copy the contents of new_k and new_v back to k and v
        cudaMemcpy(k, new_k, B * NH * T * HS * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v, new_v, B * NH * T * HS * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(new_k);
        cudaFree(new_v);
    }

    // size_t k1size = sizeof(k) / sizeof(k[0]);
    // size_t v1size = sizeof(v) / sizeof(v[0]);
    // printf("%ld, %ld", k1size, v1size);

    // Batched matrix multiply with cuBLAS for QK^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));
    // size_t sizepatt = sizeof(preatt) / sizeof(preatt[0]);
    // printf("Preatt: %ld", sizepatt);

    // Multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);

    cudaCheck(cudaGetLastError());

    // New approach: first cuBLAS another batched matmul
    float *vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // Now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}