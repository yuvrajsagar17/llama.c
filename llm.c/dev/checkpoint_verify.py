import numpy as np
def checkpoint_verify(file_path, expected_param_sizes):
    with open(file_path, "rb") as f:
        # Read and verify the header
        header = np.fromfile(f, dtype=np.int32, count=256)
        magic_number = header[0]
        version = header[1]
        max_seq_len = header[2]
        vocab_size = header[3]
        num_layers = header[4]
        num_heads = header[5]
        channels = header[6]
        padded_vocab_size = header[7]
        
        if magic_number != 20240326:
            print("Bad magic number in model file")
            return
        if version != 3:
            print("Bad version in model file")
            return
        
        print(f"max_seq_len: {max_seq_len}")
        print(f"vocab_size: {vocab_size}")
        print(f"num_layers: {num_layers}")
        print(f"num_heads: {num_heads}")
        print(f"channels: {channels}")
        print(f"padded_vocab_size: {padded_vocab_size}")
        
        # Verify parameter sizes
        for size in expected_param_sizes:
            param = np.fromfile(f, dtype=np.float32, count=size)
            if len(param) != size:
                print(f"Expected size {size}, but got {len(param)}")
                return

        print("Checkpoint verification passed!")



if __name__== '__main__':
    # Define the expected parameter sizes based on the new structure
    max_seq_len = 1024
    vocab_size = 50257
    num_layers = 12
    num_heads = 12
    channels = 768
    padded_vocab_size = 50304

    expected_param_sizes = [
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
    checkpoint_verify(file_path, expected_param_sizes)