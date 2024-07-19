#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include "common.h"

__global__ void repeat_interleave_forward_kernel(float *dst, const float *src, int B, int num_kv_heads, int T, int HS, int queries_per_kv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * num_kv_heads * queries_per_kv * T * HS;

    if (idx < total_threads)
    {
        int b = idx / (num_kv_heads * queries_per_kv * T * HS);
        int rest = idx % (num_kv_heads * queries_per_kv * T * HS);
        int nh = rest / (T * HS);
        rest = rest % (T * HS);
        int t = rest / HS;
        int hs = rest % HS;

        // Map destination head index to source head index
        int src_nh = nh % num_kv_heads;
        int src_idx = (b * num_kv_heads * T * HS) + (src_nh * T * HS) + (t * HS) + hs;
        int dst_idx = idx;
        dst[dst_idx] = src[src_idx];
    }
}

__global__ void repeat_interleave_backward_kernel(float *dsrc, const float *ddst, int B, int num_kv_heads, int T, int HS, int queries_per_kv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * num_kv_heads * queries_per_kv * T * HS;

    if (idx < total_threads)
    {
        int b = idx / (num_kv_heads * queries_per_kv * T * HS);
        int rest = idx % (num_kv_heads * queries_per_kv * T * HS);
        int nh = rest / (T * HS);
        rest = rest % (T * HS);
        int t = rest / HS;
        int hs = rest % HS;

        int src_nh = nh % num_kv_heads;
        int src_idx = (b * num_kv_heads * T * HS) + (src_nh * T * HS) + (t * HS) + hs;
        atomicAdd(&dsrc[src_idx], ddst[idx]);
    }
}

void attention_forward_gqa(float *out, float *qkvr, float *att, float *inp,
                           int B, int T, int C, int NH, int num_kv_heads)
{
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH;              // head size
    int kv_HS = C / num_kv_heads; // key/value head size
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
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

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

        cudaFree(k);
        cudaFree(v);
        k = new_k;
        v = new_v;
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

void attention_backward_gqa(float *dinp, float *dqkvr, float *dpreatt, float *datt, float *scratch,
                            const float *dout,
                            const float *qkvr, const float *att,
                            int B, int T, int C, int NH, int num_kv_heads)
{
    const int block_size = 256;
    int HS = C / NH; // head size
    int queries_per_kv = NH / num_kv_heads;
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));

    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));

    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());

    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));

    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));

    // // Allocate intermediate tensors for backward repeat interleave
    // float *dsrc_k, *dsrc_v;
    // cudaMalloc((void **)&dsrc_k, B * num_kv_heads * *T * HS * sizeof(float));
    // cudaMalloc((void **)&dsrc_v, B * num_kv_heads * T * HS * sizeof(float));
    // cudaMemset(dsrc_k, 0, B * num_kv_heads * T * HS * sizeof(float));
    // cudaMemset(dsrc_v, 0, B * num_kv_heads * T * HS * sizeof(float));

    // // backward through repeat interleave operation for dk and dv
    // num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    // repeat_interleave_backward_kernel<<<num_blocks, block_size>>>(dsrc_k, dk, B, NH, T, HS, num_kv_heads);
    // repeat_interleave_backward_kernel<<<num_blocks, block_size>>>(dsrc_v, dv, B, NH, T, HS, num_kv_heads);
    // cudaCheck(cudaGetLastError());

    // Repeat interleave for GQA if num_kv_heads != NH
    if (num_kv_heads != NH)
    {
        // Allocate intermediate tensors for backward repeat interleave
        float *dsrc_k, *dsrc_v;
        cudaMalloc((void **)&dsrc_k, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));
        cudaMalloc((void **)&dsrc_v, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));
        cudaMemset(dsrc_k, 0, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));
        cudaMemset(dsrc_v, 0, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));

        // backward through repeat interleave operation for dk and dv
        int repeat_interleave_threads = B * NH * T * HS;
        num_blocks = CEIL_DIV(repeat_interleave_threads, block_size);
        repeat_interleave_backward_kernel<<<num_blocks, block_size>>>(dsrc_k, dk, B, num_kv_heads, T, HS, queries_per_kv);
        repeat_interleave_backward_kernel<<<num_blocks, block_size>>>(dsrc_v, dv, B, num_kv_heads, T, HS, queries_per_kv);
        cudaCheck(cudaGetLastError());

        // backward into inp
        num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
        permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dsrc_k, dsrc_v, B, T, NH, HS);
        cudaCheck(cudaGetLastError());

        // Cleanup
        cudaFree(dsrc_k);
        cudaFree(dsrc_v);
    }
    else
    {
        // backward into inp
        // backward into inp without repeat interleave
        num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
        permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }
}
