#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void swiglu_backward_cpu(float *dinp, const float *inp, const float *gate, const float *dout, int N)
{
    /**
     * y=SiLU(xW)*(xV)
     * z=xW
     * g=xV
     *
     * Using Chain-Rule
     * ∂y/∂x = (σ(z) + z*σ(z)*(1−σ(z)))*g*W + SiLU(z)*V
     */
    for (int i = 0; i < N; i++)
    {
        float z = inp[i];
        float g = gate[i];
        float y = z / (1.0f + expf(-z)) * g;                   // SwiGLU(x)
        float sig_z = 1.0f / (1.0f + expf(-z));                // Sigmoid(z)
        float silu_prime = sig_z + z * sig_z * (1.0f - sig_z); // SiLU'(xW)
        float grad_z = (silu_prime * g) * dout[i];             // Gradient w.r.t. xW
        float grad_g = (z / (1.0f + expf(-z))) * dout[i];      // Gradient w.r.t. xV
        dinp[i] = grad_z + grad_g;                             // Sum of gradients
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void swiglu_backward_kernel1(floatX *dinp, const floatX *inp, const floatX *gate, const floatX *dout, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float xW = (float)inp[i];
        float xV = (float)gate[i];
        float y = xW / (1.0f + expf(-xW)) * xV;                    // SwiGLU(x)
        float sig_xW = 1.0f / (1.0f + expf(-xW));                  // Sigmoid(xW)
        float silu_prime = sig_xW + xW * sig_xW * (1.0f - sig_xW); // SiLU'(xW)
        float grad_xW = (silu_prime * xV) * dout[i];               // Gradient w.r.t. xW
        float grad_xV = (xW / (1.0f + expf(-xW))) * dout[i];       // Gradient w.r.t. xV
        dinp[i] = grad_xW + grad_xV;                               // Sum of gradients
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void swiglu_backward1(floatX *dinp, const floatX *inp, const floatX *gate, const floatX *dout, int N, const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    swiglu_backward_kernel1<<<grid_size, block_size>>>(dinp, inp, gate, dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void swiglu_backward(int kernel_num,
                     floatX *dinp,
                     const floatX *inp,
                     const floatX *gate,
                     const floatX *dout,
                     int B, int T, int C,
                     int block_size)
{
    switch (kernel_num)
    {
    case 1:
        swiglu_backward1(dinp, inp, gate, dout, B * T * C, block_size);
        break;

    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float *dinp = (float *)malloc(B * T * C * sizeof(float));
    float *inp = make_random_float(B * T * C);  // precomputed x*W
    float *gate = make_random_float(B * T * C); // precomputed x*V
    float *dout = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    swiglu_backward_cpu(dinp, inp, gate, dout, B * T * C);

    // move to GPU
    floatX *d_dinp;
    floatX *d_inp;
    floatX *d_gate;
    floatX *d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_gate, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_gate, gate, B * T * C));
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        swiglu_backward(kernel_num, d_dinp, d_inp, d_gate, d_dout, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, swiglu_backward,
                                              kernel_num, d_dinp, d_inp, d_gate, d_dout,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * (int)sizeof(floatX); // Include dout in memory operations
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(dinp);
    free(inp);
    free(gate);
    free(dout);

    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_gate));
    cudaCheck(cudaFree(d_dout));
    return 0;
}
