void attention_forward_gqa(float *out, float *qkvr, float *att, float *inp,
                           int B, int T, int C, int NH, int num_kv_heads, cublasHandle_t cublas_handle)
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
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // Batched matrix multiply with cuBLAS for QK^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, kv_HS, &alpha, k, kv_HS, T * kv_HS, q, HS, T * HS, &beta, preatt, T, T * T, B * num_kv_heads));

    // Multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // New approach: first cuBLAS another batched matmul
    float *vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, kv_HS, T, T, &alpha, v, kv_HS, T * kv_HS, att, T, T * T, &beta, vaccum, kv_HS, T * kv_HS, B * num_kv_heads));

    // Now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// --------------------------------------------------------------------------------------------------------------------------------
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <assert.h>
#include <cooperative_groups.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void repeat_interleave_kernel(float *dst, const float *src, int B, int NH, int T, int HS, int num_kv_heads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * HS;

    if (idx < total_threads)
    {
        int b = idx / (NH * T * HS);
        int rest = idx % (NH * T * HS);
        int nh = rest / (T * HS);
        rest = rest % (T * HS);
        int t = rest / HS;
        int hs = rest % HS;

        // Calculate source head index, alternating between the original heads
        int src_nh = (nh / 2) + (nh % 2) * (num_kv_heads / 2);
        int src_idx = (b * num_kv_heads * T * HS) + (src_nh * T * HS) + (t * HS) + hs;
        dst[idx] = src[src_idx];
    }
}

void attention_forward_gqa(float *out, float *qkvr, float *att, float *inp,
                           int B, int T, int C, int NH, int num_kv_heads, cublasHandle_t cublas_handle)
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
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // Repeat interleave for GQA
    if (num_kv_heads != NH)
    {
        float *new_k, *new_v;
        cudaMalloc((void **)&new_k, B * NH * T * kv_HS * sizeof(float));
        cudaMalloc((void **)&new_v, B * NH * T * kv_HS * sizeof(float));

        repeat_interleave_kernel<<<num_blocks, block_size>>>(new_k, k, B, NH, T, kv_HS, queries_per_kv);
        repeat_interleave_kernel<<<num_blocks, block_size>>>(new_v, v, B, NH, T, kv_HS, queries_per_kv);
        cudaCheck(cudaGetLastError());

        cudaFree(k);
        cudaFree(v);
        k = new_k;
        v = new_v;
    }

    // Batched matrix multiply with cuBLAS for QK^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, kv_HS, &alpha, k, kv_HS, T * kv_HS, q, HS, T * HS, &beta, preatt, T, T * T, B * num_kv_heads));

    // Multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // New approach: first cuBLAS another batched matmul
    float *vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, kv_HS, T, T, &alpha, v, kv_HS, T * kv_HS, att, T, T * T, &beta, vaccum, kv_HS, T * kv_HS, B * num_kv_heads));

    // Now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

int main()
{
    // Assuming necessary initializations and input allocations are done here
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;
    int num_kv_heads = 6;

    float *inp = ...      // Initialize inp
        float *qkvr = ... // Initialize qkvr
        float *att = ...  // Initialize att
        float *out = ...  // Initialize out

        attention_forward_gqa(out, qkvr, att, inp, B, T, C, NH, num_kv_heads, cublas_handle);

    // Cleanup
    cublasDestroy(cublas_handle);
    // Free other allocated memory
    return 0;
}
