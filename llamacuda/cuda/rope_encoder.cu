/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_forward.cu -o encoder_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./encoder_forward 1

version 2 is more optimized, parallelizes over all of B,T,C
./encoder_forward 2

version 3 is like version 2 but uses float4 reads/writes
./encoder_forward 3
*/

// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include <cassert>

// #define ENABLE_BF16
// #include "common.h"

// // ----------------------------------------------------------------------------
// // CPU code reference
// // GPT-2 positional encoder forward pass
// void encoder_forward_cpu(float *out,
//                          const int *inp, const float *wte, const float *wpe,
//                          int B, int T, int C)
// {
//     for (int b = 0; b < B; b++)
//     {
//         for (int t = 0; t < T; t++)
//         {
//             float *out_bt = out + b * T * C + t * C;
//             int ix = inp[b * T + t];
//             const float *wte_ix = wte + ix * C;
//             const float *wpe_t = wpe + t * C;
//             for (int i = 0; i < C; i++)
//             {
//                 out_bt[i] = wte_ix[i] + wpe_t[i];
//             }
//         }
//     }
// }
// // ----------------------------------------------------------------------------
// // GPU kernels
// // naive implementation into kernel, parallelize over B,T, loop over C
// __global__ void encoder_forward_kernel1(floatX *out,
//                                         const int *inp, const floatX *wte, const floatX *wpe,
//                                         int B, int T, int C)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int N = B * T;
//     if (idx < N)
//     {
//         int b = idx / T;
//         int t = idx % T;
//         floatX *out_bt = out + b * T * C + t * C;
//         int ix = inp[b * T + t];
//         const floatX *wte_ix = wte + ix * C;
//         const floatX *wpe_t = wpe + t * C;
//         for (int i = 0; i < C; i++)
//         {
//             out_bt[i] = (floatX)((float)wte_ix[i] + (float)wpe_t[i]);
//         }
//     }
// }
// // optimized implementation: parallelize over all of B,T,C
// __global__ void encoder_forward_kernel2(floatX *out,
//                                         const int *inp, const floatX *wte, const floatX *wpe,
//                                         int B, int T, int C)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int N = B * T * C;
//     if (idx < N)
//     {
//         int bt = idx / C;
//         int b = bt / T;
//         int t = bt % T;
//         int c = idx % C;
//         int ix = inp[b * T + t];
//         floatX *out_btc = out + b * T * C + t * C + c;
//         const floatX *wte_ix = wte + ix * C + c;
//         const floatX *wpe_tc = wpe + t * C + c;
//         *out_btc = (floatX)((float)*wte_ix + (float)*wpe_tc);
//     }
// }
// __global__ void encoder_forward_kernel3(floatX *out,
//                                         const int *inp, const floatX *wte, const floatX *wpe,
//                                         int B, int T, int C)
// {
//     int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
//     int N = B * T * C;
//     if (idx < N)
//     {
//         int bt = idx / C;
//         int b = bt / T;
//         int t = bt % T;
//         int c = idx % C;
//         int ix = inp[b * T + t];
//         floatX *out_btc = out + b * T * C + t * C + c;
//         const floatX *wte_ix = wte + ix * C + c;
//         const floatX *wpe_tc = wpe + t * C + c;
//         x128 packed_out;
//         x128 wte = load128cs(wte_ix);
//         x128 wpe = load128cs(wpe_tc);
// #pragma unroll
//         for (int k = 0; k < wte.size; k++)
//         {
//             packed_out[k] = (floatX)((float)wte[k] + (float)wpe[k]);
//         }
//         store128(out_btc, packed_out);
//     }
// }
// // ----------------------------------------------------------------------------
// // kernel launcher
// void encoder_forward1(floatX *out,
//                       const int *inp, const floatX *wte, const floatX *wpe,
//                       int B, int T, int C,
//                       const int block_size)
// {
//     const int N = B * T;
//     const int grid_size = ceil_div(N, block_size);
//     encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
//     cudaCheck(cudaGetLastError());
// }
// void encoder_forward2(floatX *out,
//                       const int *inp, const floatX *wte, const floatX *wpe,
//                       int B, int T, int C,
//                       const int block_size)
// {
//     const int N = B * T * C;
//     const int grid_size = ceil_div(N, block_size);
//     encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
//     cudaCheck(cudaGetLastError());
// }
// void encoder_forward3(floatX *out,
//                       const int *inp, const floatX *wte, const floatX *wpe,
//                       int B, int T, int C,
//                       const int block_size)
// {
//     const int N = B * T * C;
//     const int grid_size = ceil_div(N, (int)(block_size * x128::size));
//     encoder_forward_kernel3<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
//     cudaCheck(cudaGetLastError());
// }
// // kernel version dispatch
// void encoder_forward(int kernel_num,
//                      floatX *out,
//                      const int *inp, const floatX *wte, const floatX *wpe,
//                      int B, int T, int C,
//                      const int block_size)
// {
//     switch (kernel_num)
//     {
//     case 1:
//         encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
//         break;
//     case 2:
//         encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
//         break;
//     case 3:
//         encoder_forward3(out, inp, wte, wpe, B, T, C, block_size);
//         break;
//     default:
//         printf("Invalid kernel number\n");
//         exit(1);
//     }
// }
// // ----------------------------------------------------------------------------
// int main(int argc, char **argv)
// {
//     setup_main();
//     int B = 8;
//     int T = 1024;
//     int C = 768;
//     int V = 50257;
//     int deviceIdx = 0;
//     cudaCheck(cudaSetDevice(deviceIdx));
//     // create host memory of random numbers
//     float *out = (float *)malloc(B * T * C * sizeof(float));
//     int *inp = make_random_int(B * T, V);
//     float *wte = make_random_float(V * C);
//     float *wpe = make_random_float(T * C);
//     // move to GPU
//     floatX *d_out;
//     int *d_inp;
//     floatX *d_wte;
//     floatX *d_wpe;
//     cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
//     cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
//     cudaCheck(cudaMalloc(&d_wte, V * C * sizeof(floatX)));
//     cudaCheck(cudaMalloc(&d_wpe, T * C * sizeof(floatX)));
//     cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));
//     cudaCheck(memcpy_convert(d_wte, wte, V * C));
//     cudaCheck(memcpy_convert(d_wpe, wpe, T * C));
//     // read kernel_num from command line
//     int kernel_num = 2;
//     if (argc > 1)
//     {
//         kernel_num = atoi(argv[1]);
//     }
//     printf("Using kernel %d\n", kernel_num);
//     // first check the correctness of the kernel
//     encoder_forward_cpu(out, inp, wte, wpe, B, T, C);
//     // time the kernel at different block sizes
//     int block_sizes[] = {32, 64, 128, 256, 512, 1024};
//     for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
//     {
//         int block_size = block_sizes[j];
//         printf("Checking block size %d.\n", block_size);
//         encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
// #if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
//         float tol = 1e-5;
// #else
//         float tol = 1e-2f;
// #endif
//         validate_result(d_out, out, "out", B * T * C, tol);
//     }
//     printf("All results match. Starting benchmarks.\n\n");
//     for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
//     {
//         int block_size = block_sizes[j];
//         int repeat_times = 1000;
//         float elapsed_time = benchmark_kernel(repeat_times, encoder_forward,
//                                               kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
//         // napkin math: estimate the memory bandwidth achieved
//         // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
//         // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
//         long memory_ops = B * T * C * 4 * 4;
//         float memory_bandwidth = memory_ops / elapsed_time / 1e6;
//         printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
//     }
//     // free memory
//     free(out);
//     free(inp);
//     free(wte);
//     free(wpe);
//     cudaCheck(cudaFree(d_out));
//     cudaCheck(cudaFree(d_inp));
//     cudaCheck(cudaFree(d_wte));
//     cudaCheck(cudaFree(d_wpe));
//     return 0;
// }

// ----------------------------------------------------------------------------------------------------------------------------------------

/**
 * Starting to write RoPE (Roatary Position Embeddings).
 * Reference: Paper (https://arxiv.org/pdf/2104.09864)
 *
 * Helping Material: https://medium.com/@zaiinn440/linear-rope-vs-ntk-vs-yarn-vs-cope-d33587ddfd35 by Zain ul Abideen
 *
 */

#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

__global__ void precompute_freqs_cis_kernel(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dim / 2)
    {
        float freq = 1.0f / powf(theta, (float)tid * 2.0f / dim); // float powf(float base, float exponent);

        for (int t = 0; t < end; t++)
        {
            freqs_cos[t * (dim / 2) + tid] = cosf(t * freq);
            freqs_sin[t * (dim / 2) + tid] = sinf(t * freq);
        }
    }
}

void precompute_freqs_cis_forward(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    int threads = 64;
    int blocks = (dim / 2 + threads - 1) / threads;
    precompute_freqs_cis_kernel<<<blocks, threads>>>(freqs_cos, freqs_sin, dim, end, theta);
    cudaDeviceSynchronize();
}

__global__ void apply_rope_kernel(
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

void apply_rope(float *q, float *k, float *freqs_cos, float *freqs_sin, int B, int T, int NH, int head_dim)
{
    dim3 blocks(B, T, NH);
    int threads = head_dim / 2;
    apply_rope_kernel<<<blocks, threads>>>(q, k, freqs_cos, freqs_sin, B, T, NH, head_dim);
    cudaDeviceSynchronize();
}

int main()
{
    // Model parameters
    int B = 8;    // Batch size
    int T = 1024; // Sequence length
    int NH = 12;  // Number of heads
    int C = 768;  // Model dimension (Embedding Features)

    int dim = C / NH; // head_dim
    float theta = 10000.0f;

    // Allocate memory for frequencies
    float *freqs_cos;
    float *freqs_sin;
    cudaMalloc(&freqs_cos, T * (dim / 2) * sizeof(float));
    cudaMalloc(&freqs_sin, T * (dim / 2) * sizeof(float));

    // Precompute frequencies
    precompute_freqs_cis(freqs_cos, freqs_sin, dim, T * 2, theta);

    // Allocate memory for q and k matrices
    float *q;
    float *k;
    cudaMalloc(&q, B * T * NH * dim * sizeof(float));
    cudaMalloc(&k, B * T * NH * dim * sizeof(float));

    // Initialize q and k matrices with some values (for example purposes)
    float *h_q = new float[B * T * NH * dim];
    float *h_k = new float[B * T * NH * dim];
    for (int i = 0; i < B * T * NH * dim; ++i)
    {
        h_q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_k[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy q and k matrices to GPU
    cudaMemcpy(q, h_q, B * T * NH * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(k, h_k, B * T * NH * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Apply RoPE
    apply_rope(q, k, freqs_cos, freqs_sin, B, T, NH, dim);

    // Copy results back to host
    cudaMemcpy(h_q, q, B * T * NH * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k, k, B * T * NH * dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_q;
    delete[] h_k;
    cudaFree(freqs_cos);
    cudaFree(freqs_sin);
    cudaFree(q);
    cudaFree(k);

    return 0;
}
