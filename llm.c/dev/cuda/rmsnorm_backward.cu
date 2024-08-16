/*
Kernels for layernorm backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_backward.cu -o layernorm_backward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_backward 1

version 2 moves a lot of reduction to shared memory over global memory
./layernorm_backward 2
*/

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

// LLama RMSNorm forward pass
void rmsnorm_forward_cpu(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C)
{
    float eps = 1e-5f;
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // seek to the input position inp[b,t,:]
            const float *x = inp + b * T * C + t * C;
            // calculate the rms (root mean square)
            float rms = 0.0f;
            for (int i = 0; i < C; i++)
            {
                rms += x[i] * x[i];
            }
            rms = sqrtf(rms / C + eps);
            // seek to the output position in out[b,t,:]
            float *out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++)
            {
                float n = x[i] / rms;              // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o;                     // write
            }
        }
    }
}

void rmsnorm_backward_cpu(float *dinp, float *dweight, float *dbias,
                          const float *dout, const float *inp, const float *weight, const float *bias,
                          int B, int T, int C)
{
    float eps = 1e-5f;
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            const float *dout_bt = dout + b * T * C + t * C;
            const float *inp_bt = inp + b * T * C + t * C;
            float *dinp_bt = dinp + b * T * C + t * C;

            // Calculate the rms
            float rms = 0.0f;
            for (int i = 0; i < C; i++)
            {
                rms += inp_bt[i] * inp_bt[i];
            }
            rms = sqrtf(rms / C + eps);

            // First, calculate the gradients for the weights and biases
            for (int i = 0; i < C; i++)
            {
                float norm = inp_bt[i] / rms;
                dbias[i] += dout_bt[i];
                dweight[i] += norm * dout_bt[i];
            }

            // Now, calculate the gradients for the inputs
            float drms = 0.0f;
            for (int i = 0; i < C; i++)
            {
                drms += inp_bt[i] * dout_bt[i] * weight[i];
            }
            drms = drms * (-1.0f / (rms * rms * rms * C));

            for (int i = 0; i < C; i++)
            {
                float norm = inp_bt[i] / rms;
                dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// GPU helper functions for atomicAdd on smaller than 32-bit types
#ifdef ENABLE_BF16
__device__ void atomicAddX(__nv_bfloat16 *addr, __nv_bfloat16 val)
{
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    __nv_bfloat162 *ptr_bf16 = reinterpret_cast<__nv_bfloat162 *>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    __nv_bfloat162 add_val = (ptr_val & 0x3) ? __halves2bfloat162(__ushort_as_bfloat16(0), val)
                                             : __halves2bfloat162(val, __ushort_as_bfloat16(0));
    atomicAdd(ptr_bf16, add_val);
}
#endif
#ifdef ENABLE_FP16
__device__ void atomicAddX(half *addr, half val)
{
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    half2 *ptr_fp16 = reinterpret_cast<half2 *>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    half2 add_val = (ptr_val & 0x3) ? __halves2half2(__ushort_as_half(0), val)
                                    : __halves2half2(val, __ushort_as_half(0));
    atomicAdd(ptr_fp16, add_val);
}
#endif
__device__ void atomicAddX(float *addr, float val)
{
    atomicAdd(addr, val);
}

__global__ void rmsnorm_backward_kernel1(float *dinp, float *dweight, float *dbias,
                                         const float *dout, const float *inp, const float *weight, const float *bias,
                                         int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float eps = 1e-5f;
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;

    // Calculate the rms
    float rms = 0.0f;
    for (int i = 0; i < C; i++)
    {
        rms += inp_bt[i] * inp_bt[i];
    }
    rms = sqrtf(rms / C + eps);

    // First, calculate the gradients for the weights and biases
    for (int i = 0; i < C; i++)
    {
        float norm = inp_bt[i] / rms;
        atomicAdd(&dbias[i], dout_bt[i]);
        atomicAdd(&dweight[i], norm * dout_bt[i]);
    }

    // Calculate drms
    float drms = 0.0f;
    for (int i = 0; i < C; i++)
    {
        drms += inp_bt[i] * dout_bt[i] * weight[i];
    }
    drms = drms * (-1.0f / (rms * rms * rms * C));

    // Now, calculate the gradients for the inputs
    for (int i = 0; i < C; i++)
    {
        float norm = inp_bt[i] / rms;
        dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void rmsnorm_backward1(float *dinp, float *dweight, float *dbias,
                       const float *dout, const float *inp, const float *weight, const float *bias,
                       int B, int T, int C, const int block_size)
{
    const int N = B * T;
    const int grid_size = (N + block_size - 1) / block_size; // equivalent to ceil(N / block_size)
    rmsnorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void rmsnorm_backward(int kernel_num,
                      floatX *dinp, floatX *dweight, floatX *dbias,
                      const floatX *dout, const floatX *inp, const floatX *weight, const floatX *bias,
                      int B, int T, int C,
                      const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        rmsnorm_backward1(dinp, dweight, dbias, dout, inp, weight, bias, B, T, C, block_size);
        break;

    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 1600; // this is the problematic size

    // first do the forward pass in CPU
    float *out = (float *)malloc(B * T * C * sizeof(float));

    float *inp = make_random_float(B * T * C);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);

    rmsnorm_forward_cpu(out, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    float *dbias = make_zeros_float(C);

    rmsnorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, bias, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    floatX *d_dinp;
    floatX *d_dweight;
    floatX *d_dbias;
    floatX *d_dout;
    floatX *d_inp;
    floatX *d_weight;
    floatX *d_bias;

    float *d_scratch;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dbias, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_scratch, (1024 / 32) * cuda_num_SMs * (2 * C + 1) * sizeof(float)));
    // copy over the "inputs" to the backward call
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_weight, weight, C));
    cudaCheck(memcpy_convert(d_bias, bias, C));

    // launch the kernel
    // removed 768 because it doesn't work for kernel9 despite being OK in train_gpt2.cu?!
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dbias, 0, C * sizeof(floatX)));

        rmsnorm_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_bias,
                         B, T, C, block_size);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f;    // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        printf("dbias:\n");
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_backward, kernel_num,
                                              d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                                              B, T, C, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(out);

    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(d_scratch));
    return 0;
}
