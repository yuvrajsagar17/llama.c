/*
Kernels for layernorm forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt rmsnorm_forward.cu -o rmsnorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./rmsnorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4

verstion 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
./layernorm_forward 5
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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

// // GPT-2 layernorm forward pass
// void layernorm_forward_cpu(float *out, float *mean, float *rstd,
//                            const float *inp, const float *weight, const float *bias,
//                            int B, int T, int C)
// {
//     float eps = 1e-5f;
//     for (int b = 0; b < B; b++)
//     {
//         for (int t = 0; t < T; t++)
//         {
//             // seek to the input position inp[b,t,:]
//             const float *x = inp + b * T * C + t * C;
//             // calculate the mean
//             float m = 0.0f;
//             for (int i = 0; i < C; i++)
//             {
//                 m += x[i];
//             }
//             m = m / C;
//             // calculate the variance (without any bias correction)
//             float v = 0.0f;
//             for (int i = 0; i < C; i++)
//             {
//                 float xshift = x[i] - m;
//                 v += xshift * xshift;
//             }
//             v = v / C;
//             // calculate the rstd
//             float s = 1.0f / sqrtf(v + eps);
//             // seek to the output position in out[b,t,:]
//             float *out_bt = out + b * T * C + t * C;
//             for (int i = 0; i < C; i++)
//             {
//                 float n = (s * (x[i] - m));        // normalized output
//                 float o = n * weight[i] + bias[i]; // scale and shift it
//                 out_bt[i] = o;                     // write
//             }
//             // cache the mean and rstd for the backward pass later
//             mean[b * T + t] = m;
//             rstd[b * T + t] = s;
//         }
//     }
// }

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void rmsnorm_forward_kernel1(float *out, const float *inp, const float *weight, const float *bias, int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N)
    {
        // seek to the input position inp[idx,:]
        const float *x = inp + idx * C;
        // calculate the rms (root mean square)
        float rms = 0.0f;
        for (int i = 0; i < C; i++)
        {
            rms += x[i] * x[i];
        }
        rms = sqrtf(rms / C + eps);
        // seek to the output position in out[idx,:]
        float *out_idx = out + idx * C;
        for (int i = 0; i < C; i++)
        {
            float n = x[i] / rms;              // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o;                    // write
        }
    }
}

// __global__ void mean_kernel(float *mean, const float *inp, int N, int C, int block_size)
// {
//     extern __shared__ float shared[];
//     int idx = blockIdx.x;  // range [0, B*T)
//     int tid = threadIdx.x; // range [0, block_size)
//     const float *x = inp + idx * C;
//     // thread coarsening
//     float sum = 0.0f;
//     for (int i = tid; i < C; i += block_size)
//     {
//         sum += x[i];
//     }
//     shared[tid] = sum;
//     __syncthreads();
//     // reductions
//     for (int stride = block_size / 2; stride >= 1; stride /= 2)
//     {
//         __syncthreads();
//         if (tid < stride)
//         {
//             shared[tid] += shared[tid + stride];
//         }
//     }
//     // write the final result (at thread 0) to global memory
//     if (tid == 0)
//     {
//         mean[idx] = shared[0] / C;
//     }
// }

// __global__ void rstd_kernel(float *rstd, const float *inp, const float *mean, int N, int C, int block_size)
// {
//     extern __shared__ float shared[];
//     int idx = blockIdx.x;  // range [0, B*T)
//     int tid = threadIdx.x; // range [0, block_size)
//     const float *x = inp + idx * C;
//     float m = mean[idx];
//     // thread coarsening
//     float sum = 0.0f;
//     for (int i = tid; i < C; i += block_size)
//     {
//         float diff = x[i] - m;
//         sum += diff * diff;
//     }
//     shared[tid] = sum;
//     __syncthreads();
//     // reductions
//     for (int stride = block_size / 2; stride >= 1; stride /= 2)
//     {
//         __syncthreads();
//         if (tid < stride)
//         {
//             shared[tid] += shared[tid + stride];
//         }
//     }
//     // write the final result (at thread 0) to global memory
//     if (tid == 0)
//     {
//         rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
//     }
// }

// __global__ void normalization_kernel(float *out, const float *inp, float *mean, float *rstd,
//                                      const float *weight, const float *bias, int B, int T, int C)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     int bt = idx / C;
//     int c = idx % C;

//     float m = mean[bt];
//     float s = rstd[bt];
//     float xi = inp[idx];
//     float n = s * (xi - m);
//     float o = n * weight[c] + bias[c];

//     out[idx] = o;
// }

// ----------------------------------------------------------------------------
// kernel launcher

void rmsnorm_forward1(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, const int block_size)
{
    const int N = B * T;
    const int grid_size = (N + block_size - 1) / block_size; // equivalent to ceil(N / block_size)
    rmsnorm_forward_kernel1<<<grid_size, block_size>>>(out, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// void rmsnorm_forward2(float *out, float *mean, float *rstd,
//                       const float *inp, const float *weight, const float *bias,
//                       int B, int T, int C,
//                       const int block_size)
// {
//     int N = B * T;
//     // in mean and rstd, threads cooperate within blocks via reductions
//     mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
//     cudaCheck(cudaGetLastError());
//     rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
//     cudaCheck(cudaGetLastError());
//     // in the normalization, everything just gets flattened out
//     const int block_size2 = 256;
//     const int grid_size = ceil_div(B * T * C, block_size2);
//     normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
//     cudaCheck(cudaGetLastError());
// }

// kernel version dispatch
void rmsnorm_forward(int kernel_num,
                     float *out,
                     const float *inp, const float *weight, const float *bias,
                     int B, int T, int C,
                     const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        rmsnorm_forward1(out, inp, weight, bias, B, T, C, block_size);
        break;
    // case 2:
    //     rmsnorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
    //     break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float *out = (float *)malloc(B * T * C * sizeof(float));
    // float *mean = (float *)malloc(B * T * sizeof(float));
    // float *rstd = (float *)malloc(B * T * sizeof(float));
    float *inp = make_random_float(B * T * C);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);

    // move to GPU
    float *d_out;
    // float *d_mean;
    // float *d_rstd;
    float *d_inp;
    float *d_weight;
    float *d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    // cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    // cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rmsnorm_forward_cpu(out, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        rmsnorm_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // // time the kernel at different block sizes
    // for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    // {
    //     int block_size = block_sizes[j];

    //     int repeat_times = 2000;
    //     float elapsed_time = benchmark_kernel(repeat_times, rmsnorm_forward,
    //                                           kernel_num, d_out, d_inp, d_weight, d_bias,
    //                                           B, T, C, block_size);

    //     // napkin math: estimate the memory bandwidth achieved
    //     // e.g. A100 40GB PCIe is advertised at 1,555GB/s
    //     long memory_ops = (2 * B * T * C) * 4; // *4 for float
    //     float memory_bandwidth = memory_ops / elapsed_time / 1e6;

    //     printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
}

// free memory
free(out);
// free(mean);
// free(rstd);
free(inp);
free(weight);
free(bias);
cudaCheck(cudaFree(d_out));
// cudaCheck(cudaFree(d_mean));
// cudaCheck(cudaFree(d_rstd));
cudaCheck(cudaFree(d_inp));
cudaCheck(cudaFree(d_weight));
cudaCheck(cudaFree(d_bias));

return 0;
}