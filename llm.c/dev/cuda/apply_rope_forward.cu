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

__global__ void apply_rope_forward_kernel(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int NH, int HS)
{

    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    int half_hs = HS / 2;

    if (hs < half_hs)
    {
        int index = b * T * NH * HS + t * NH * HS + nh * HS + hs;
        int freq_index = t * half_hs + hs;

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        float q_r = q[index];
        float q_i = q[index + half_hs];
        float k_r = k[index];
        float k_i = k[index + half_hs];

        q[index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
        q[index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i

        k[index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
}

void apply_rope_forward(float *q, float *k, float *freqs_cos, float *freqs_sin, int B, int T, int NH, int HS)
{
    dim3 blocks(B, T, NH);
    int threads = HS / 2;
    apply_rope_forward_kernel<<<blocks, threads>>>(q, k, freqs_cos, freqs_sin, B, T, NH, HS);
    cudaDeviceSynchronize();
}