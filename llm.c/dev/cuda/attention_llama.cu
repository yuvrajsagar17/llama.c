/*
Kernels for attention forward pass.

If you do not have CUDNN, you can remove ENABLE_CUDNN to run the other kernels

See the README for cuDNN install instructions

Compile example with cuDNN:
nvcc -I/PATH/TO/cudnn-frontend/include -DENABLE_CUDNN -O3 --use_fast_math --lcublas -lcublasLt -lcudnn attention_forward.cu -o attention_forward

Compile example without cuDNN:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_llama.cu -o attention_llama
./attention_llama

*/
// #define ENABLE_CUDNN // can be enabled via nvcc "-DENABLE_CUDNN"
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
// CPU code reference

void attention_forward_cpu(float *out, float *preatt, float *att,
                           const float *inp,
                           int B, int T, int C, int NH, int num_kv_heads)
{
    // Implementing GQA
    if (NH % num_kv_heads != 0)
    {
        return;
    }

    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C * 3;
    int hs = C / NH;              // Head size for queries
    int kv_hs = C / num_kv_heads; // Head size for keys and values
    float scale = 1.0 / sqrtf(hs);
    int queries_per_kv = NH / num_kv_heads;

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                // Determine the appropriate kv head based on the query head
                int kv_h = (num_kv_heads == NH) ? h : h / queries_per_kv;

                const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;

                // Calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    const float *key_t2 = inp + b * T * C3 + t2 * C3 + kv_h * kv_hs + C; // +C because it's key

                    float val = 0.0f;
                    for (int i = 0; i < kv_hs; i++)
                    {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval)
                    {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }

                // Pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t + 1; t2 < T; t2++)
                {
                    preatt_bth[t2] = -INFINITY;
                }

                // Calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // Normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++)
                {
                    if (t2 <= t)
                    {
                        att_bth[t2] *= expsum_inv;
                    }
                    else
                    {
                        att_bth[t2] = 0.0f;
                    }
                }

                // Accumulate weighted values into the output of attention
                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++)
                {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++)
                {
                    const float *value_t2 = inp + b * T * C3 + t2 * C3 + kv_h * kv_hs + 2 * C; // +2*C for value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < kv_hs; i++)
                    {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
__global__ void attention_query_key_kernel1(float *preatt, const float *inp,
                                            int B, int T, int C, int NH, int num_kv_heads, int queries_per_kv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;
    int hs = C / NH;
    int kv_hs = C / num_kv_heads;
    float scale = 1.0 / sqrtf(hs);

    if (idx < total_threads)
    {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t)
        {
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int kv_h = (num_kv_heads == NH) ? h : h / queries_per_kv;

        const float *query_t = inp + b * T * C * 3 + t * C * 3 + h * hs;
        const float *key_t2 = inp + b * T * C * 3 + t2 * C * 3 + kv_h * kv_hs + C;

        float val = 0.0f;
        for (int i = 0; i < kv_hs; i++)
        {
            val += query_t[i] * key_t2[i];
        }
        val *= scale;

        preatt[idx] = val;
    }
}

__global__ void attention_softmax_kernel1(float *att, const float *preatt,
                                          int B, int T, int NH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads)
    {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        const float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++)
        {
            if (preatt_bth[t2] > maxval)
            {
                maxval = preatt_bth[t2];
            }
        }

        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++)
        {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        for (int t2 = 0; t2 < T; t2++)
        {
            if (t2 <= t)
            {
                att_bth[t2] *= expsum_inv;
            }
            else
            {
                att_bth[t2] = 0.0f;
            }
        }
    }
}

__global__ void attention_value_kernel1(float *out, const float *att, const float *inp,
                                        int B, int T, int C, int NH, int num_kv_heads, int queries_per_kv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;
    int hs = C / NH;
    int kv_hs = C / num_kv_heads;

    if (idx < total_threads)
    {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int kv_h = (num_kv_heads == NH) ? h : h / queries_per_kv;

        float *out_bth = out + b * T * C + t * C + h * hs;
        const float *att_bth = att + b * NH * T * T + h * T * T + t * T;

        for (int i = 0; i < hs; i++)
        {
            out_bth[i] = 0.0f;
        }
        for (int t2 = 0; t2 <= t; t2++)
        {
            const float *value_t2 = inp + b * T * C * 3 + t2 * C * 3 + kv_h * kv_hs + 2 * C;
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < kv_hs; i++)
            {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

__global__ void permute_kernel_backward(float *dinp,
                                        const float *dq, const float *dk, const float *dv,
                                        int B, int N, int NH, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH, int d)
{
    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = __ldcs(&inp[idx]);
    }
}

__global__ void unpermute_kernel_backward(float *dinp, const float *dout, int B, int N, int NH, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

__device__ float &vec_at(float4 &vec, int index)
{
    return reinterpret_cast<float *>(&vec)[index];
}

__device__ float vec_at(const float4 &vec, int index)
{
    return reinterpret_cast<const float *>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float *out, float inv_temperature, const float *inp, int N, int T)
{
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4 == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank(); // backward order
    if (idx >= N * T)
    {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float *x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4 *x_vec = reinterpret_cast<const float4 *>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size())
    {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for (int k = 0; k < 4; ++k)
        {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for (int k = 0; k < 4; ++k)
        {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if (4 * pos_by_4 + warp.thread_rank() <= own_pos)
    {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4 * pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4 * pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size())
    {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

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

// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward1(float *out, float *preatt, float *att,
                        const float *inp,
                        int B, int T, int C, int NH, int num_kv_heads,
                        const int block_size)
{
    int queries_per_kv = NH / num_kv_heads;

    int total_threads = B * NH * T * T;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH, num_kv_heads, queries_per_kv);

    total_threads = B * T * NH;
    num_blocks = (total_threads + block_size - 1) / block_size;
    attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH, num_kv_heads, queries_per_kv);
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

// kernel version dispatch
void attention_forward(int kernel_num,
                       float *out, float *stats, float *vaccum,
                       float *qkvr, float *preatt, float *att,
                       float *inp,
                       int B, int T, int C, int NH, , int num_kv_heads,
                       const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        attention_forward1(out, preatt, att, inp, B, T, C, NH, num_kv_heads, block_size);
        break;

    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}
// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;
    int num_kv_heads = 6; // No. of kv heads

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS (and cuDNN if needed)
    cublasCreate(&cublas_handle);
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

#ifdef ENABLE_CUDNN
    checkCudnnErr(cudnnCreate(&cudnn_handle));
#endif

    // create host memory of random numbers
    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *preatt = (float *)malloc(B * NH * T * T * sizeof(float));
    float *att = (float *)malloc(B * NH * T * T * sizeof(float));
    // float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float *inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float *d_out;
    float *d_stats; // for cuDNN
    float *d_vaccum;
    float *d_qkvr;
    float *d_preatt;
    float *d_att;
    float *d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_stats, B * NH * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);
    int block_sizes[] = {32, 64, 128, 256, 512};

    // Lower accuracy requirements for FP16 (1e-4f also too much for TF32 on kernels 3 & 4)
    float accuracy_threshold = (kernel_num <= 4) ? 1e-3f : 1e-2f;

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH, num_kv_heads);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        attention_forward(kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, num_kv_heads, block_size);
        // all kernels should produce the correct output out
        // todo - make accuracy threshold dynamic and depend on FP16 vs FP32?
        validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
        // but as for preatt and att, things get a bit more complicated:
        if (kernel_num != 2 && kernel_num < 5)
        {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
            validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
        }
        if (kernel_num != 2 && kernel_num < 4)
        {
            // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
            // into the softmax, so preatt is off by 1.0f / sqrt(HS)
            // but att and out (checked below) should match.
            validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);
        }
    }
    printf("All results match. Starting benchmarks.\n\n");
    first_run_validation = false;

    // benchmark speed of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                              kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att,
                                              d_inp, B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    cublasDestroy(cublas_handle);

#ifdef ENABLE_CUDNN
    cudnnDestroy(cudnn_handle);
    if (cudnn_workspace_size > 0)
    {
        cudaCheck(cudaFree(cudnn_workspace));
    }
#endif

    return 0;
}