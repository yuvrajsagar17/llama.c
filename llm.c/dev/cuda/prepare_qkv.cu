// #include "matmul_forward.cu"

// void prepare_qkv(float *inp, const float *weight, const float *bias,
//                  int B, int T, int C, int NH, int num_kv_heads)
// {
//     int C3 = C * 3;
//     int hs = C / NH;              // Head size for queries
//     int kv_hs = C / num_kv_heads; // Head size for keys and values

//     float *combined_out = (float *)malloc(B * T * C3 * sizeof(float));
//     matmul_forward_cpu(combined_out, inp, weight, bias, B, T, C, C3);

// #pragma omp parallel for collapse(2)
//     for (int b = 0; b < B; b++)
//     {
//         for (int t = 0; t < T; t++)
//         {
//             float *q_bt = inp + b * T * C3 + t * C3;         // Start of Q
//             float *k_bt = inp + b * T * C3 + t * C3 + C;     // Start of K
//             float *v_bt = inp + b * T * C3 + t * C3 + 2 * C; // Start of V
//             const float *combined_bt = combined_out + b * T * C3 + t * C3;

//             for (int h = 0; h < NH; h++)
//             {
//                 for (int i = 0; i < hs; i++)
//                 {
//                     q_bt[h * hs + i] = combined_bt[h * hs + i];
//                 }
//             }

//             for (int h = 0; h < num_kv_heads; h++)
//             {
//                 for (int i = 0; i < kv_hs; i++)
//                 {
//                     k_bt[h * kv_hs + i] = combined_bt[C + h * kv_hs + i];
//                     v_bt[h * kv_hs + i] = combined_bt[2 * C + h * kv_hs + i];
//                 }
//             }
//         }
//     }

//     free(combined_out);
// }
