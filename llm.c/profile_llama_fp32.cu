/*
This code is a convenience tool for profiling the CUDA kernels in the training
loop of train_llama_fp32.cu. Compile:

make profile_llamafp32cu

And then e.g. use ncu from NVIDIA. The CLI docs for example:
https://docs.nvidia.com/nsight-compute/NsightComputeCli/

TLDR run like:

sudo ncu --set full --import-source yes -o profile -f ./profile_llamafp32cu

This:
- `--set full` means we'll collect A LOT of metrics. take out for less
- `--import-source yes` means we'll get the source code in the profile
- `-o profile` writes the results into file profile.ncu-rep
- `-f` forces overwrite of the profile.ncu-rep file
- `./profile_llamafp32cu` is the executable we want to profile

This writes results into profile.ncu-rep output file.
You can open this up in NVIDIA Nsight Compute UI.
For example, I have NVIDIA Nsight Compute installed on my Mac, and I rsync
the profile.ncu-rep from a cloud box to local to pretty view.
*/

#define TESTING
#include "train_llama_fp32.cu"

int main(int argc, char *argv[])
{

    LLaMA3 model;
    load_model_params(&model); // Assuming load_model_params loads your model's parameters

    int B = 24;   // if program OOMs decrease this number, e.g. all the way down to 4 or etc
    int T = 1024; // if even that OOMs move on to this one. keep them nice and powers of 2
    printf("batch size: %d\n", B);
    printf("sequence length: %d\n", T);

    int *x = (int *)mallocCheck(B * T * sizeof(int));
    int *y = (int *)mallocCheck(B * T * sizeof(int));
    for (int i = 0; i < B * T; ++i)
    {
        x[i] = i % model.config.vocab_size;
        y[i] = i % model.config.vocab_size;
    }

    // Override number of layers to 1 because all layers repeat the same kernels, only profile once
    model.config.num_layers = 1;

    // Do a training step
    llama3_forward(&model, x, y, B, T); // Forward pass
    llama3_backward(&model);            // Backward pass

    // Update model parameters using AdamW optimizer
    llama3_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, 1); // Update step

    cudaCheck(cudaDeviceSynchronize()); // Finish all CUDA work to get correct precise timings

    // Free resources
    llama3_free(&model);
    return 0;
}
