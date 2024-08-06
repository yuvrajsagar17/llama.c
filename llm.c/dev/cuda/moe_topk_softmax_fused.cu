#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to perform top-k selection and softmax
__global__ void topk_softmax_kernel(const float *scores, float *top_values, int *top_indices, int rows, int cols, int k)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < rows)
    {
        extern __shared__ float shared_memory[];
        float *shared_vals = shared_memory;
        int *shared_indices = (int *)&shared_memory[cols];

        // Load scores into shared memory
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
        {
            shared_vals[i] = scores[row_idx * cols + i];
            shared_indices[i] = i;
        }
        __syncthreads();

        // Simple selection sort to find top-k values
        for (int i = 0; i < k; ++i)
        {
            int max_idx = i;
            for (int j = i + 1; j < cols; ++j)
            {
                if (shared_vals[j] > shared_vals[max_idx])
                {
                    max_idx = j;
                }
            }
            // Swap values
            float temp_val = shared_vals[i];
            shared_vals[i] = shared_vals[max_idx];
            shared_vals[max_idx] = temp_val;

            // Swap indices
            int temp_idx = shared_indices[i];
            shared_indices[i] = shared_indices[max_idx];
            shared_indices[max_idx] = temp_idx;
        }

        // Compute softmax on top-k values
        float max_val = shared_vals[0];
        float sum_exp = 0.0;
        for (int i = 0; i < k; ++i)
        {
            shared_vals[i] = expf(shared_vals[i] - max_val);
            sum_exp += shared_vals[i];
        }
        for (int i = 0; i < k; ++i)
        {
            shared_vals[i] /= sum_exp;
        }

        // Write top-k values (softmax) and indices to global memory
        for (int i = 0; i < k; ++i)
        {
            top_values[row_idx * k + i] = shared_vals[i];
            top_indices[row_idx * k + i] = shared_indices[i];
        }
    }
}

// Function to call the kernel
void topk_softmax(const float *d_scores, float *d_top_values, int *d_top_indices, int batch_size, int seq_len, int num_experts, int k)
{
    int rows = batch_size * seq_len;
    int cols = num_experts;
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;

    size_t shared_memory_size = 2 * cols * sizeof(float);

    topk_softmax_kernel<<<grid_size, block_size, shared_memory_size>>>(d_scores, d_top_values, d_top_indices, rows, cols, k);
    cudaDeviceSynchronize();
}

int main()
{
    int batch_size = 8;
    int seq_len = 1024;
    int num_experts = 4;
    int k = 2;

    int rows = batch_size * seq_len;
    int cols = num_experts;

    // Allocate and initialize host memory
    float *h_scores = (float *)malloc(rows * cols * sizeof(float));
    // Fill h_scores with example data
    for (int i = 0; i < rows * cols; ++i)
    {
        h_scores[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_scores;
    float *d_top_values;
    int *d_top_indices;
    cudaMalloc((void **)&d_scores, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_top_values, rows * k * sizeof(float));
    cudaMalloc((void **)&d_top_indices, rows * k * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_scores, h_scores, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Call top-k softmax function
    topk_softmax(d_scores, d_top_values, d_top_indices, batch_size, seq_len, num_experts, k);

    // Copy results back to host
    float *h_top_values = (float *)malloc(rows * k * sizeof(float));
    int *h_top_indices = (int *)malloc(rows * k * sizeof(int));
    cudaMemcpy(h_top_values, d_top_values, rows * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_indices, d_top_indices, rows * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the results
    for (int i = 0; i < rows; ++i)
    {
        printf("Row %d:\n", i);
        for (int j = 0; j < k; ++j)
        {
            printf("  Top value (softmax): %f, index: %d\n", h_top_values[i * k + j], h_top_indices[i * k + j]);
        }
    }

    // Free memory
    free(h_scores);
    free(h_top_values);
    free(h_top_indices);
    cudaFree(d_scores);
    cudaFree(d_top_values);
    cudaFree(d_top_indices);

    return 0;
}
