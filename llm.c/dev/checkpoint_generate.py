import numpy as np

def checkpoint_generate(file_path, param_sizes):
    with open(file_path, "wb") as f:
        # Write the header
        magic_number = 20240326
        version = 3
        max_seq_len = 1024
        vocab_size = 50257
        num_layers = 12
        num_heads = 12
        channels = 768
        padded_vocab_size = 50304
        header = np.array([magic_number, version, max_seq_len, vocab_size, num_layers, num_heads, channels, padded_vocab_size] + [0]*248, dtype=np.int32)
        f.write(header.tobytes())
        
        # Write the parameters
        for size in param_sizes:
            param = np.random.rand(size).astype(np.float32)  # Replace with actual parameter values
            f.write(param.tobytes())


if __name__ == '__main__':

    # Define the expected parameter sizes based on the new structure
    max_seq_len = 1024
    vocab_size = 50257
    num_layers = 12
    num_heads = 12
    channels = 768
    padded_vocab_size = 50304

    # Example parameter sizes based on the new structure
    param_sizes = [
        vocab_size * channels,           # wte
        max_seq_len * channels,          # wpe
        num_layers * channels,           # ln1w
        num_layers * channels,           # ln1b
        num_layers * (3 * channels) * channels,  # qkvw
        num_layers * (3 * channels),     # qkvb
        num_layers * channels * channels,        # attprojw
        num_layers * channels,           # attprojb
        num_layers * channels,           # ln2w
        num_layers * channels,           # ln2b
        num_layers * (4 * channels) * channels,  # fcw
        num_layers * (4 * channels),     # fcb
        num_layers * (4 * channels) * channels,  # fcw_g
        num_layers * (4 * channels),     # fcb_g
        num_layers * channels * (4 * channels),  # fcprojw
        num_layers * channels,           # fcprojb
        channels,                        # lnfw
        channels                         # lnfb
    ]

    file_path = '../gpt2_124M.bin'
    checkpoint_generate(file_path, param_sizes)