// Xavier Initialization
void xavier_initialization(float *param_values, size_t size, size_t fan_in, size_t fan_out)
{
    float scale = sqrt(2.0 / (fan_in + fan_out));
    for (size_t i = 0; i < size; i++)
    {
        param_values[i] = scale * ((float)rand() / RAND_MAX - 0.5);
    }
}

void load_model_params(GPT2 *model)
{
    // Initialize model configuration and parameters from given param_values
    model->config.max_seq_len = 1024;
    model->config.vocab_size = 50257;
    model->config.num_layers = 12;
    model->config.num_heads = 12;
    model->config.channels = 768;
    model->config.padded_vocab_size = 50304;

    // allocate space for all the parameters and point them to the right places
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        size_t size = model->param_sizes[i];
        size_t fan_in = (i == 0) ? model->config.padded_vocab_size : model->config.channels;
        size_t fan_out = (i == 0) ? model->config.channels : model->config.channels;
        xavier_initialization(model->params_memory + i * size, size, fan_in, fan_out);
    }

    // Copy parameters to device
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize activations
    fill_in_activation_sizes(model->act_sizes, 8, 1024, model->config);
    // count the number of activation tensors
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        num_activations += model->act_sizes[i];
    }
    model->num_activations = num_activations;

    model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

    model->acts_memory = (float *)mallocCheck(model->num_activations * sizeof(float));
    cudaCheck(cudaMalloc((void **)&model->acts_memory, model->num_activations * sizeof(float)));

    // Initialize gradients
    model->grads_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
    cudaCheck(cudaMalloc((void **)&model->grads_memory, model->num_parameters * sizeof(float)));

    // Initialize AdamW optimizer buffers
    model->m_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
    model->v_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
    cudaCheck(cudaMalloc((void **)&model->m_memory, model->num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&model->v_memory, model->num_parameters * sizeof(float)));

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}