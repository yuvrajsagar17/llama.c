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

__global__ void apply_rope_backward_kernel(
    float *dq, float *dk, const float *q, const float *k,
    const float *freqs_cos, const float *freqs_sin,
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

        // Gradients with respect to q and k (already computed)
        float dq_r = dq[index];
        float dq_i = dq[index + half_hs];
        float dk_r = dk[index];
        float dk_i = dk[index + half_hs];

        // Gradients with respect to q and k
        dq[index] = dq_r * cos_val + dq_i * sin_val;
        dq[index + half_hs] = dq_i * cos_val - dq_r * sin_val;
        dk[index] = dk_r * cos_val + dk_i * sin_val;
        dk[index + half_hs] = dk_i * cos_val - dk_r * sin_val;

        /**
         * WE DON'T NEED TO ACCUMULATE THE GRADIENTs IN d_freq_cos and d_freq_sin.
         */
        // // Gradients with respect to freqs_cos and freqs_sin
        // float d_freq_cos_q = q_r * dq_r + q_i * dq_i;
        // float d_freq_sin_q = -q_i * dq_r + q_r * dq_i;
        // float d_freq_cos_k = k_r * dk_r + k_i * dk_i;
        // float d_freq_sin_k = -k_i * dk_r + k_r * dk_i;

        // atomicAdd(&d_freq_cos[freq_index], d_freq_cos_q + d_freq_cos_k);
        // atomicAdd(&d_freq_sin[freq_index], d_freq_sin_q + d_freq_sin_k);
    }
}

void apply_rope_backward(
    float *dq, float *dk, const float *q, const float *k,
    const float *freqs_cos, const float *freqs_sin,
    int B, int T, int NH, int HS)
{
    dim3 blocks(B, T, NH);
    dim3 threads(HS / 2);
    apply_rope_backward_kernel<<<blocks, threads>>>(
        dq, dk, q, k, freqs_cos, freqs_sin, B, T, NH, HS);
    cudaDeviceSynchronize();
}
