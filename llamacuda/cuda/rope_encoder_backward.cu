#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

__global__ void apply_rope_backward_kernel(
    float *dq, float *dk, const float *q, const float *k,
    const float *freqs_cos, const float *freqs_sin,
    float *d_freq_cos, float *d_freq_sin,
    const float *dout_q, const float *dout_k,
    int B, int T, int NH, int C)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int c = threadIdx.x;

    int half_c = C / 2;

    if (c < half_c)
    {
        int index = b * T * NH * C + t * NH * C + nh * C + c;
        int freq_index = t * half_c + c;

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        float q_r = q[index];
        float q_i = q[index + half_c];
        float k_r = k[index];
        float k_i = k[index + half_c];

        // Gradients from the output
        float dq_r = dout_q[index];
        float dq_i = dout_q[index + half_c];
        float dk_r = dout_k[index];
        float dk_i = dout_k[index + half_c];

        // Gradients with respect to q and k
        dq[index] = dq_r * cos_val + dq_i * sin_val;
        dq[index + half_c] = dq_i * cos_val - dq_r * sin_val;
        dk[index] = dk_r * cos_val + dk_i * sin_val;
        dk[index + half_c] = dk_i * cos_val - dk_r * sin_val;

        // Gradients with respect to freqs_cos and freqs_sin
        float d_freq_cos_q = q_r * dq_r + q_i * dq_i;
        float d_freq_sin_q = -q_i * dq_r + q_r * dq_i;
        float d_freq_cos_k = k_r * dk_r + k_i * dk_i;
        float d_freq_sin_k = -k_i * dk_r + k_r * dk_i;

        atomicAdd(&d_freq_cos[freq_index], d_freq_cos_q + d_freq_cos_k);
        atomicAdd(&d_freq_sin[freq_index], d_freq_sin_q + d_freq_sin_k);
    }
}

void apply_rope_backward(
    float *dq, float *dk, const float *q, const float *k,
    const float *freqs_cos, const float *freqs_sin,
    float *d_freq_cos, float *d_freq_sin,
    const float *dout_q, const float *dout_k,
    int B, int T, int NH, int C)
{
    dim3 blocks(B, T, NH);
    int threads = C / 2;
    apply_rope_backward_kernel<<<blocks, threads>>>(
        dq, dk, q, k, freqs_cos, freqs_sin, d_freq_cos, d_freq_sin, dout_q, dout_k, B, T, NH, C);
    cudaDeviceSynchronize();
}

__global__ void precompute_freqs_cis_backward_kernel(
    float *d_freqs_cos, float *d_freqs_sin,
    const float *freqs_cos, const float *freqs_sin,
    const float *d_freq_cos, const float *d_freq_sin,
    int dim, int end, float theta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dim / 2)
    {
        float freq = 1.0f / powf(theta, (float)tid * 2.0f / dim);

        for (int t = 0; t < end; t++)
        {
            int index = t * (dim / 2) + tid;

            // Values of cos and sin
            float cos_val = cosf(t * freq);
            float sin_val = sinf(t * freq);

            // Gradients from the output
            float d_out_cos = d_freq_cos[index];
            float d_out_sin = d_freq_sin[index];

            // Chain rule: propagate the gradients
            float d_freq = d_out_cos * (-sin_val) + d_out_sin * cos_val;

            // Accumulate the gradients for freqs_cos and freqs_sin
            atomicAdd(&d_freqs_cos[index], d_out_cos * cos_val);
            atomicAdd(&d_freqs_sin[index], d_out_sin * sin_val);
        }
    }
}

void precompute_freqs_cis_backward(
    float *d_freqs_cos, float *d_freqs_sin,
    const float *freqs_cos, const float *freqs_sin,
    const float *d_freq_cos, const float *d_freq_sin,
    int dim, int end, float theta)
{
    int threads = 64;
    int blocks = (dim / 2 + threads - 1) / threads;
    precompute_freqs_cis_backward_kernel<<<blocks, threads>>>(
        d_freqs_cos, d_freqs_sin, freqs_cos, freqs_sin, d_freq_cos, d_freq_sin, dim, end, theta);
    cudaDeviceSynchronize();
}

int main()
{
    // Model parameters
    int B = 2;  // Batch size
    int T = 3;  // Sequence length
    int NH = 4; // Number of heads
    int C = 8;  // Model dimension

    int dim = C / NH;
    int freq_size = T * (dim / 2);
    float theta = 10000.0f;

    // Allocate memory on host
    float *h_q = new float[B * T * NH * C];
    float *h_k = new float[B * T * NH * C];
    float *h_dq = new float[B * T * NH * C];
    float *h_dk = new float[B * T * NH * C];
    float *h_dout_q = new float[B * T * NH * C];
    float *h_dout_k = new float[B * T * NH * C];
    float *h_freqs_cos = new float[freq_size];
    float *h_freqs_sin = new float[freq_size];
    float *h_d_freq_cos = new float[freq_size];
    float *h_d_freq_sin = new float[freq_size];
    float *h_d_freqs_cos = new float[freq_size];
    float *h_d_freqs_sin = new float[freq_size];

    // Initialize host memory (example data)
    for (int i = 0; i < B * T * NH * C; ++i)
    {
        h_q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_k[i] = static_cast<float>(rand()) / RAND_MAX;
        h_dout_q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_dout_k[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < freq_size; ++i)
    {
        h_freqs_cos[i] = static_cast<float>(rand()) / RAND_MAX;
        h_freqs_sin[i] = static_cast<float>(rand()) / RAND_MAX;
        h_d_freq_cos[i] = 0.0f;
        h_d_freq_sin[i] = 0.0f;
        h_d_freqs_cos[i] = 0.0f;
        h_d_freqs_sin[i] = 0.0f;
    }

    // Allocate memory on device
    float *d_q, *d_k, *d_dq, *d_dk, *d_dout_q, *d_dout_k;
    float *d_freqs_cos, *d_freqs_sin, *d_d_freq_cos, *d_d_freq_sin, *d_d_freqs_cos, *d_d_freqs_sin;
    cudaMalloc(&d_q, B * T * NH * C * sizeof(float));
    cudaMalloc(&d_k, B * T * NH * C * sizeof(float));
    cudaMalloc(&d_dq, B * T * NH * C * sizeof(float));
    cudaMalloc(&d_dk, B * T * NH * C * sizeof(float));
    cudaMalloc(&d_dout_q, B * T * NH * C * sizeof(float));
    cudaMalloc(&d_dout_k, B * T * NH * C * sizeof(float));
    cudaMalloc(&d_freqs_cos, freq_size * sizeof(float));
    cudaMalloc(&d_freqs_sin, freq_size * sizeof(float));
    cudaMalloc(&d_d_freq_cos, freq_size * sizeof(float));
    cudaMalloc(&d_d_freq_sin, freq_size * sizeof(float));
    cudaMalloc(&d_d_freqs_cos, freq_size * sizeof(float));
    cudaMalloc(&d_d_freqs_sin, freq_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_q, h_q, B * T * NH * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, B * T * NH * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout_q, h_dout_q, B * T * NH * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout_k, h_dout_k, B * T * NH * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freqs_cos, h_freqs_cos, freq_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freqs_sin, h_freqs_sin, freq_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_freq_cos, h_d_freq_cos, freq_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_freq_sin, h_d_freq_sin, freq_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_freqs_cos, h_d_freqs_cos, freq_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_freqs_sin, h_d_freqs_sin, freq_size * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke the apply_rope_backward kernel
    apply_rope_backward(d_dq, d_dk, d_q, d_k, d_freqs_cos, d_freqs_sin, d_d_freq_cos, d_d_freq_sin, d_dout_q, d_dout_k, B, T, NH, C);

    // Invoke the precompute_freqs_cis_backward kernel
    precompute_freqs_cis_backward(d_d_freqs_cos, d_d_freqs_sin, d_freqs_cos, d_freqs_sin, d_d_freq_cos, d_d_freq_sin, dim, T, theta);

    // Copy results back to host
    cudaMemcpy(h_dq, d_dq, B * T * NH * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dk, d_dk, B * T * NH * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_freqs_cos, d_d_freqs_cos, freq_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_freqs_sin, d_d_freqs_sin, freq_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_q;
    delete[] h_k;
    delete[] h_dq;
    delete[] h_dk;
    delete[] h_dout_q;
    delete[] h_dout_k;
    delete[] h_freqs_cos;
    delete[] h_freqs_sin;
    delete[] h_d_freq_cos;
    delete[] h_d_freq_sin;
    delete[] h_d_freqs_cos;
    delete[] h_d_freqs_sin;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dout_q);
    cudaFree(d_dout_k);
    cudaFree(d_freqs_cos);
    cudaFree(d_freqs_sin);
    cudaFree(d_d_freq_cos);
    cudaFree(d_d_freq_sin);
    cudaFree(d_d_freqs_cos);
    cudaFree(d_d_freqs_sin);

    return 0;
}

__global__ void precompute_freqs_cis_backward_kernel(
    float *d_freqs_cos, float *d_freqs_sin,
    const float *freqs_cos, const float *freqs_sin,
    int dim, int end, float theta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dim / 2)
    {
        float freq = 1.0f / powf(theta, (float)tid * 2.0f / dim);

        for (int t = 0; t < end; t++)
        {
            int index = t * (dim / 2) + tid;

            // Values of cos and sin
            float cos_val = cosf(t * freq);
            float sin_val = sinf(t * freq);

            // Gradients from the output
            float d_out_cos = d_freq_cos[index];
            float d_out_sin = d_freq_sin[index];

            // Chain rule: propagate the gradients
            float d_freq = d_out_cos * (-sin_val) + d_out_sin * cos_val;

            // Accumulate the gradients for freqs_cos and freqs_sin
            atomicAdd(&d_freqs_cos[index], d_out_cos * cos_val);
            atomicAdd(&d_freqs_sin[index], d_out_sin * sin_val);
        }
    }
}

void precompute_freqs_cis_backward(
    float *d_freqs_cos, float *d_freqs_sin,
    const float *freqs_cos, const float *freqs_sin,
    int dim, int end, float theta)
{
    int threads = 64;
    int blocks = (dim / 2 + threads - 1) / threads;
    precompute_freqs_cis_backward_kernel<<<blocks, threads>>>(
        d_freqs_cos, d_freqs_sin, freqs_cos, freqs_sin, dim, end, theta);
    cudaDeviceSynchronize();
}