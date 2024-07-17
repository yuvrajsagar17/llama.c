/*
Kernels for attention forward pass.

If you do not have CUDNN, you can remove ENABLE_CUDNN to run the other kernels

See the README for cuDNN install instructions

Compile example with cuDNN:
nvcc -I/PATH/TO/cudnn-frontend/include -DENABLE_CUDNN -O3 --use_fast_math --lcublas -lcublasLt -lcudnn attention_forward.cu -o attention_forward

Compile example without cuDNN:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
./attention_forward 1

version 2 is a naive implementation of flash attention, taken, adapted from
https://github.com/tspeterkim/flash-attention-minimal
and with help from
https://github.com/leloykun/flash-hyperbolic-attention-minimal
sadly, this flash attention version seems about 3X slower than the naive version
./attention_forward 2

version 3 is a cuBLAS + softmax version, similar to the PyTorch implementation
cuBLAS is used both to calculate the QK^T and the final weighted sum
the softmax is calculated using a custom, efficient kernel as well
this turns out to be ~20X faster than (1) nice
./attention_forward 3

version 4 is a further optimized kernel that fuses the scale operation,
uses a directly autoregressive softmax, and uses the online softmax algorithm.
./attention_forward 4

version 5 is a FP16 version of kernel 4
./attention_forward 5

version 6 is kernel 5 skipping (un)permute (unrealistic but useful comparison point)

version 10 is using cuDNN Flash Attention using FP16 or BF16, see:
https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md
./attention_forward 10

version 11 is kernel 10 skipping FP16/FP32 conversions (full FP16/BF16 network)
./attention_forward 11
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