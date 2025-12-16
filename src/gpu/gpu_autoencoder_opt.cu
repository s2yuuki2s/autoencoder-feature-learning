#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "gpu/gpu_autoencoder_opt.h"
#include "gpu/gpu_layers.h"
#include "cpu/cpu_layers.h"
#include "constants.h"
#include "data_loader.h"

#define CHECK(call)                                                                                 \
    do                                                                                              \
    {                                                                                               \
        cudaError_t err = call;                                                                     \
        if (err != cudaSuccess)                                                                     \
        {                                                                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

static float rand_uniform(std::mt19937 &rng, float a, float b)
{
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

// Helper: Cấp phát từ Pool
static float *allocate_from_pool(float *pool_base, size_t &current_offset, int count)
{
    float *ptr = pool_base + current_offset;
    current_offset += count;
    return ptr;
}

// ================== KERNEL DEFINITIONS ==================

__global__ void sgd_update_kernel_opt(float *w, const float *dw, float lr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    w[idx] -= lr * dw[idx];
}

__global__ void mse_grad_kernel_opt(const float *__restrict__ recon,
                                    const float *__restrict__ target,
                                    float *__restrict__ grad_recon,
                                    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    grad_recon[idx] = (2.0f / (float)total) * (recon[idx] - target[idx]);
}

// --- Backward Kernels Definitions ---
// Định nghĩa trực tiếp tại đây để tránh lỗi Linker

__global__ void conv2d_backward_kernel_opt(
    const float *__restrict__ in, const float *__restrict__ grad_out, const float *__restrict__ W,
    int n, int width, int height, int in_c, int out_c,
    float *__restrict__ grad_in, float *__restrict__ grad_W, float *__restrict__ grad_b)
{
    const int KH = 3, KW = 3;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * height * width * out_c;
    if (idx >= total)
        return;

    int f = idx % out_c;
    int tmp = idx / out_c;
    int j = tmp % width;
    tmp /= width;
    int i = tmp % height;
    int img = tmp / height;

    const float *grad_out_pix = grad_out + img * width * height * out_c + (i * width + j) * out_c;
    float go = grad_out_pix[f];

    atomicAdd(&grad_b[f], go);

    for (int fi = 0; fi < KH; ++fi)
    {
        int row = i + fi - KH / 2;
        if (row < 0 || row >= height)
            continue;
        for (int fj = 0; fj < KW; ++fj)
        {
            int col = j + fj - KW / 2;
            if (col < 0 || col >= width)
                continue;

            const float *in_pix = in + img * width * height * in_c + (row * width + col) * in_c;
            const float *w_ptr = W + f * KH * KW * in_c + (fi * KW + fj) * in_c;

            float *grad_in_pix = grad_in + img * width * height * in_c + (row * width + col) * in_c;
            float *grad_w_ptr = grad_W + f * KH * KW * in_c + (fi * KW + fj) * in_c;

            for (int c = 0; c < in_c; ++c)
            {
                atomicAdd(&grad_w_ptr[c], go * in_pix[c]);
                atomicAdd(&grad_in_pix[c], go * w_ptr[c]);
            }
        }
    }
}

__global__ void relu_backward_kernel_opt(
    const float *__restrict__ in, const float *__restrict__ grad_out,
    float *__restrict__ grad_in, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    grad_in[idx] = (in[idx] > 0.0f) ? grad_out[idx] : 0.0f;
}

__global__ void maxpool2x2_backward_kernel_opt(
    const float *__restrict__ in, const float *__restrict__ out, const float *__restrict__ grad_out,
    int n, int width, int height, int depth, float *__restrict__ grad_in)
{
    int out_w = width / 2;
    int out_h = height / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * out_w * out_h * depth;
    if (idx >= total)
        return;

    int c = idx % depth;
    int tmp = idx / depth;
    int j = tmp % out_w;
    tmp /= out_w;
    int i = tmp % out_h;
    int img = tmp / out_h;

    float out_val = out[idx];
    float go = grad_out[idx];

    int in_i0 = 2 * i;
    int in_j0 = 2 * j;

    const float *base_in = in + img * width * height * depth;
    float *base_grad_in = grad_in + img * width * height * depth;

    int idx00 = (in_i0 * width + in_j0) * depth + c;
    int idx01 = (in_i0 * width + (in_j0 + 1)) * depth + c;
    int idx10 = ((in_i0 + 1) * width + in_j0) * depth + c;
    int idx11 = ((in_i0 + 1) * width + (in_j0 + 1)) * depth + c;

    if (base_in[idx00] == out_val)
        atomicAdd(&base_grad_in[idx00], go);
    else if (base_in[idx01] == out_val)
        atomicAdd(&base_grad_in[idx01], go);
    else if (base_in[idx10] == out_val)
        atomicAdd(&base_grad_in[idx10], go);
    else
        atomicAdd(&base_grad_in[idx11], go);
}

__global__ void upsample2x_backward_kernel_opt(
    const float *__restrict__ grad_out, int n, int width, int height, int depth,
    float *__restrict__ grad_in)
{
    int out_w = width * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * height * width * depth;
    if (idx >= total)
        return;

    int c = idx % depth;
    int tmp = idx / depth;
    int j = tmp % width;
    tmp /= width;
    int i = tmp % height;
    int img = tmp / height;

    const float *base_grad_out = grad_out + img * (width * 2) * (height * 2) * depth;
    int idx00 = ((2 * i) * out_w + (2 * j)) * depth + c;
    int idx01 = ((2 * i) * out_w + (2 * j + 1)) * depth + c;
    int idx10 = ((2 * i + 1) * out_w + (2 * j)) * depth + c;
    int idx11 = ((2 * i + 1) * out_w + (2 * j + 1)) * depth + c;

    grad_in[idx] += base_grad_out[idx00] + base_grad_out[idx01] + base_grad_out[idx10] + base_grad_out[idx11];
}

// Kernel tính tổng bình phương sai số (Sum Squared Error)
// Dùng atomicAdd để gom kết quả lại.
// Lưu ý: atomicAdd float có thể sai số nhỏ, nhưng chấp nhận được cho training DL.
__global__ void mse_loss_kernel_opt(const float *__restrict__ recon,
                                    const float *__restrict__ target,
                                    float *__restrict__ total_loss,
                                    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride)
    {
        float diff = recon[i] - target[i];
        local_sum += diff * diff;
    }

    // Gom kết quả cục bộ vào biến toàn cục
    atomicAdd(total_loss, local_sum);
}

// ================== IMPLEMENTATION ==================

Gpu_Autoencoder_Opt::Gpu_Autoencoder_Opt(int batch_size_)
    : batch_size(batch_size_), d_memory_pool(nullptr)
{
    int B = batch_size;
    size_t total_floats = 0;

    // 1. Calculate Sizes
    int W1_s = 256 * 3 * 3 * 3;
    int W2_s = 128 * 3 * 3 * 256;
    int W3_s = 128 * 3 * 3 * 128;
    int W4_s = 256 * 3 * 3 * 128;
    int W5_s = 3 * 3 * 3 * 256;

    // Weights + Biases
    total_floats += (W1_s + 256) + (W2_s + 128) + (W3_s + 128) + (W4_s + 256) + (W5_s + 3);
    // Activations
    total_floats += (size_t)B * 32 * 32 * 256 + (size_t)B * 16 * 16 * 256 + (size_t)B * 16 * 16 * 128 + (size_t)B * 8 * 8 * 128;
    total_floats += (size_t)B * 8 * 8 * 128 + (size_t)B * 16 * 16 * 128 + (size_t)B * 16 * 16 * 256 + (size_t)B * 32 * 32 * 256;
    total_floats += (size_t)B * 32 * 32 * 3;
    total_floats += (size_t)B * 32 * 32 * 3 + (size_t)B * 32 * 32 * 3;
    // Gradients
    total_floats += (W1_s + 256) + (W2_s + 128) + (W3_s + 128) + (W4_s + 256) + (W5_s + 3);
    // Gradients Activations
    total_floats += (size_t)B * 32 * 32 * 3 + (size_t)B * 32 * 32 * 256 + (size_t)B * 16 * 16 * 256 + (size_t)B * 16 * 16 * 128;
    total_floats += (size_t)B * 8 * 8 * 128 + (size_t)B * 8 * 8 * 128 + (size_t)B * 16 * 16 * 128 + (size_t)B * 16 * 16 * 256;
    total_floats += (size_t)B * 32 * 32 * 256 + (size_t)B * 32 * 32 * 3;

    // 2. Allocate Pool
    this->total_memory_size = total_floats * sizeof(float);
    printf("[Opt] Allocating Memory Pool: %.2f MB... ", this->total_memory_size / (1024.0 * 1024.0));
    CHECK(cudaMalloc(&d_memory_pool, this->total_memory_size));
    CHECK(cudaMemset(d_memory_pool, 0, this->total_memory_size));
    printf("Done.\n");

    // 3. Map Pointers
    size_t offset = 0;
    d_W1 = allocate_from_pool(d_memory_pool, offset, W1_s);
    d_b1 = allocate_from_pool(d_memory_pool, offset, 256);
    d_W2 = allocate_from_pool(d_memory_pool, offset, W2_s);
    d_b2 = allocate_from_pool(d_memory_pool, offset, 128);
    d_W3 = allocate_from_pool(d_memory_pool, offset, W3_s);
    d_b3 = allocate_from_pool(d_memory_pool, offset, 128);
    d_W4 = allocate_from_pool(d_memory_pool, offset, W4_s);
    d_b4 = allocate_from_pool(d_memory_pool, offset, 256);
    d_W5 = allocate_from_pool(d_memory_pool, offset, W5_s);
    d_b5 = allocate_from_pool(d_memory_pool, offset, 3);

    d_h1 = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 256);
    d_p1 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 256);
    d_h2 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 128);
    d_encoded = allocate_from_pool(d_memory_pool, offset, B * 8 * 8 * 128);
    d_h3 = allocate_from_pool(d_memory_pool, offset, B * 8 * 8 * 128);
    d_u1 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 128);
    d_h4 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 256);
    d_u2 = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 256);
    d_recon = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 3);
    d_input = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 3);
    d_target = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 3);

    d_dW1 = allocate_from_pool(d_memory_pool, offset, W1_s);
    d_db1 = allocate_from_pool(d_memory_pool, offset, 256);
    d_dW2 = allocate_from_pool(d_memory_pool, offset, W2_s);
    d_db2 = allocate_from_pool(d_memory_pool, offset, 128);
    d_dW3 = allocate_from_pool(d_memory_pool, offset, W3_s);
    d_db3 = allocate_from_pool(d_memory_pool, offset, 128);
    d_dW4 = allocate_from_pool(d_memory_pool, offset, W4_s);
    d_db4 = allocate_from_pool(d_memory_pool, offset, 256);
    d_dW5 = allocate_from_pool(d_memory_pool, offset, W5_s);
    d_db5 = allocate_from_pool(d_memory_pool, offset, 3);

    d_g_recon = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 3);
    d_g_u2 = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 256);
    d_g_h4 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 256);
    d_g_u1 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 128);
    d_g_h3 = allocate_from_pool(d_memory_pool, offset, B * 8 * 8 * 128);
    d_g_encoded = allocate_from_pool(d_memory_pool, offset, B * 8 * 8 * 128);
    d_g_h2 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 128);
    d_g_p1 = allocate_from_pool(d_memory_pool, offset, B * 16 * 16 * 256);
    d_g_h1 = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 256);
    d_g_input = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 3);
    d_target = allocate_from_pool(d_memory_pool, offset, B * 32 * 32 * 3);
    d_loss_val = allocate_from_pool(d_memory_pool, offset, 1);

    recon_host = new float[B * 32 * 32 * 3];
}

Gpu_Autoencoder_Opt::~Gpu_Autoencoder_Opt()
{
    if (d_memory_pool)
        CHECK(cudaFree(d_memory_pool));
    delete[] recon_host;
}

void Gpu_Autoencoder_Opt::init_weights(int seed)
{
    std::mt19937 rng(seed);
    int W1_s = 256 * 3 * 3 * 3;
    int W2_s = 128 * 3 * 3 * 256;
    int W3_s = 128 * 3 * 3 * 128;
    int W4_s = 256 * 3 * 3 * 128;
    int W5_s = 3 * 3 * 3 * 256;

    std::vector<float> h_W1(W1_s), h_W2(W2_s), h_W3(W3_s), h_W4(W4_s), h_W5(W5_s);
    auto init_arr = [&](std::vector<float> &w, float s)
    { for(auto& v:w) v=rand_uniform(rng, -s, s); };
    init_arr(h_W1, 0.05f);
    init_arr(h_W2, 0.05f);
    init_arr(h_W3, 0.05f);
    init_arr(h_W4, 0.05f);
    init_arr(h_W5, 0.05f);

    CHECK(cudaMemcpy(d_W1, h_W1.data(), W1_s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W2, h_W2.data(), W2_s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W3, h_W3.data(), W3_s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W4, h_W4.data(), W4_s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W5, h_W5.data(), W5_s * sizeof(float), cudaMemcpyHostToDevice));
}

void Gpu_Autoencoder_Opt::forward(const float *input, int n, int width, int height, int depth)
{
    int input_size = n * 32 * 32 * 3;
    CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));

    // Call forward kernels from gpu_layers.h
    gpu_conv2D(d_input, d_W1, d_h1, n, 32, 32, 3, 256);
    gpu_add_bias(d_h1, d_b1, d_h1, n, 32, 32, 256);
    gpu_relu(d_h1, d_h1, n, 32, 32, 256);
    gpu_max_pooling(d_h1, d_p1, n, 32, 32, 256);

    gpu_conv2D(d_p1, d_W2, d_h2, n, 16, 16, 256, 128);
    gpu_add_bias(d_h2, d_b2, d_h2, n, 16, 16, 128);
    gpu_relu(d_h2, d_h2, n, 16, 16, 128);
    gpu_max_pooling(d_h2, d_encoded, n, 16, 16, 128);

    gpu_conv2D(d_encoded, d_W3, d_h3, n, 8, 8, 128, 128);
    gpu_add_bias(d_h3, d_b3, d_h3, n, 8, 8, 128);
    gpu_relu(d_h3, d_h3, n, 8, 8, 128);
    gpu_upsampling(d_h3, d_u1, n, 8, 8, 128);

    gpu_conv2D(d_u1, d_W4, d_h4, n, 16, 16, 128, 256);
    gpu_add_bias(d_h4, d_b4, d_h4, n, 16, 16, 256);
    gpu_relu(d_h4, d_h4, n, 16, 16, 256);
    gpu_upsampling(d_h4, d_u2, n, 16, 16, 256);

    gpu_conv2D(d_u2, d_W5, d_recon, n, 32, 32, 256, 3);
    gpu_add_bias(d_recon, d_b5, d_recon, n, 32, 32, 3);
}

void Gpu_Autoencoder_Opt::backward(const float *input, const float *target, int n, int width, int height, int depth)
{
    int W1_s = 256 * 3 * 3 * 3;
    int W2_s = 128 * 3 * 3 * 256;
    int W3_s = 128 * 3 * 3 * 128;
    int W4_s = 256 * 3 * 3 * 128;
    int W5_s = 3 * 3 * 3 * 256;

    // Reset Gradients
    CHECK(cudaMemset(d_dW1, 0, W1_s * sizeof(float)));
    CHECK(cudaMemset(d_db1, 0, 256 * sizeof(float)));
    CHECK(cudaMemset(d_dW2, 0, W2_s * sizeof(float)));
    CHECK(cudaMemset(d_db2, 0, 128 * sizeof(float)));
    CHECK(cudaMemset(d_dW3, 0, W3_s * sizeof(float)));
    CHECK(cudaMemset(d_db3, 0, 128 * sizeof(float)));
    CHECK(cudaMemset(d_dW4, 0, W4_s * sizeof(float)));
    CHECK(cudaMemset(d_db4, 0, 256 * sizeof(float)));
    CHECK(cudaMemset(d_dW5, 0, W5_s * sizeof(float)));
    CHECK(cudaMemset(d_db5, 0, 3 * sizeof(float)));

    CHECK(cudaMemset(d_g_recon, 0, n * 32 * 32 * 3 * sizeof(float)));
    CHECK(cudaMemset(d_g_u2, 0, n * 32 * 32 * 256 * sizeof(float)));
    CHECK(cudaMemset(d_g_h4, 0, n * 16 * 16 * 256 * sizeof(float)));
    CHECK(cudaMemset(d_g_u1, 0, n * 16 * 16 * 128 * sizeof(float)));
    CHECK(cudaMemset(d_g_h3, 0, n * 8 * 8 * 128 * sizeof(float)));
    CHECK(cudaMemset(d_g_encoded, 0, n * 8 * 8 * 128 * sizeof(float)));
    CHECK(cudaMemset(d_g_h2, 0, n * 16 * 16 * 128 * sizeof(float)));
    CHECK(cudaMemset(d_g_p1, 0, n * 16 * 16 * 256 * sizeof(float)));
    CHECK(cudaMemset(d_g_h1, 0, n * 32 * 32 * 256 * sizeof(float)));
    CHECK(cudaMemset(d_g_input, 0, n * 32 * 32 * 3 * sizeof(float)));

    int img_size = n * 32 * 32 * 3;
    CHECK(cudaMemcpy(d_input, input, img_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_target, target, img_size * sizeof(float), cudaMemcpyHostToDevice));

    // MSE Grad
    int block = 256;
    mse_grad_kernel_opt<<<(img_size + block - 1) / block, block>>>(d_recon, d_target, d_g_recon, img_size);
    CHECK(cudaDeviceSynchronize());

    // Backward using internal kernels defined above
    conv2d_backward_kernel_opt<<<(n * 32 * 32 * 3 + 255) / 256, 256>>>(d_u2, d_g_recon, d_W5, n, 32, 32, 256, 3, d_g_u2, d_dW5, d_db5);
    upsample2x_backward_kernel_opt<<<(n * 16 * 16 * 256 + 255) / 256, 256>>>(d_g_u2, n, 16, 16, 256, d_g_h4);
    relu_backward_kernel_opt<<<(n * 16 * 16 * 256 + 255) / 256, 256>>>(d_h4, d_g_h4, d_g_h4, n * 16 * 16 * 256);
    conv2d_backward_kernel_opt<<<(n * 16 * 16 * 256 + 255) / 256, 256>>>(d_u1, d_g_h4, d_W4, n, 16, 16, 128, 256, d_g_u1, d_dW4, d_db4);
    upsample2x_backward_kernel_opt<<<(n * 8 * 8 * 128 + 255) / 256, 256>>>(d_g_u1, n, 8, 8, 128, d_g_h3);
    relu_backward_kernel_opt<<<(n * 8 * 8 * 128 + 255) / 256, 256>>>(d_h3, d_g_h3, d_g_h3, n * 8 * 8 * 128);
    conv2d_backward_kernel_opt<<<(n * 8 * 8 * 128 + 255) / 256, 256>>>(d_encoded, d_g_h3, d_W3, n, 8, 8, 128, 128, d_g_encoded, d_dW3, d_db3);
    maxpool2x2_backward_kernel_opt<<<(n * 8 * 8 * 128 + 255) / 256, 256>>>(d_h2, d_encoded, d_g_encoded, n, 16, 16, 128, d_g_h2);
    relu_backward_kernel_opt<<<(n * 16 * 16 * 128 + 255) / 256, 256>>>(d_h2, d_g_h2, d_g_h2, n * 16 * 16 * 128);
    conv2d_backward_kernel_opt<<<(n * 16 * 16 * 128 + 255) / 256, 256>>>(d_p1, d_g_h2, d_W2, n, 16, 16, 256, 128, d_g_p1, d_dW2, d_db2);
    maxpool2x2_backward_kernel_opt<<<(n * 16 * 16 * 256 + 255) / 256, 256>>>(d_h1, d_p1, d_g_p1, n, 32, 32, 256, d_g_h1);
    relu_backward_kernel_opt<<<(n * 32 * 32 * 256 + 255) / 256, 256>>>(d_h1, d_g_h1, d_g_h1, n * 32 * 32 * 256);
    conv2d_backward_kernel_opt<<<(n * 32 * 32 * 256 + 255) / 256, 256>>>(d_input, d_g_h1, d_W1, n, 32, 32, 3, 256, d_g_input, d_dW1, d_db1);

    CHECK(cudaDeviceSynchronize());
}

void Gpu_Autoencoder_Opt::update_weights(float lr)
{
    int W1_s = 256 * 3 * 3 * 3;
    int W2_s = 128 * 3 * 3 * 256;
    int W3_s = 128 * 3 * 3 * 128;
    int W4_s = 256 * 3 * 3 * 128;
    int W5_s = 3 * 3 * 3 * 256;

    sgd_update_kernel_opt<<<(W1_s + 255) / 256, 256>>>(d_W1, d_dW1, lr, W1_s);
    sgd_update_kernel_opt<<<(W2_s + 255) / 256, 256>>>(d_W2, d_dW2, lr, W2_s);
    sgd_update_kernel_opt<<<(W3_s + 255) / 256, 256>>>(d_W3, d_dW3, lr, W3_s);
    sgd_update_kernel_opt<<<(W4_s + 255) / 256, 256>>>(d_W4, d_dW4, lr, W4_s);
    sgd_update_kernel_opt<<<(W5_s + 255) / 256, 256>>>(d_W5, d_dW5, lr, W5_s);

    sgd_update_kernel_opt<<<1, 256>>>(d_b1, d_db1, lr, 256);
    sgd_update_kernel_opt<<<1, 128>>>(d_b2, d_db2, lr, 128);
    sgd_update_kernel_opt<<<1, 128>>>(d_b3, d_db3, lr, 128);
    sgd_update_kernel_opt<<<1, 256>>>(d_b4, d_db4, lr, 256);
    sgd_update_kernel_opt<<<1, 32>>>(d_b5, d_db5, lr, 3);
    CHECK(cudaDeviceSynchronize());
}

void Gpu_Autoencoder_Opt::fit(const Dataset &dataset, int n_epoch, int batch_size_, float learning_rate, int seed, int checkpoint, const char *output_dir)
{
    float h_loss_val = 0.0f;

    for (int epoch = 1; epoch <= n_epoch; ++epoch)
    {
        Dataset shuffled = dataset;
        shuffle_dataset(shuffled);
        std::vector<Dataset> batches = create_minibatches(shuffled, batch_size_);
        double epoch_loss = 0.0;
        int num_batches = (int)batches.size();

        for (int b = 0; b < num_batches; ++b)
        {
            Dataset &batch = batches[b];
            int bn = batch.n;
            const float *x = batch.get_data();
            int current_batch_size = bn * 32 * 32 * 3;

            // 1. Forward (Không copy recon về nữa!)
            this->forward(x, bn, 32, 32, 3);

            // 2. Tính Loss trên GPU
            // Reset giá trị loss cũ về 0
            CHECK(cudaMemset(d_loss_val, 0, sizeof(float)));

            // Gọi kernel tính loss
            int block = 256;
            int grid = (current_batch_size + block - 1) / block;
            // Giới hạn grid size để tối ưu atomicAdd
            if (grid > 128)
                grid = 128;

            mse_loss_kernel_opt<<<grid, block>>>(d_recon, d_target, d_loss_val, current_batch_size);

            // 3. Backward & Update (Giữ nguyên)
            this->backward(x, x, bn, 32, 32, 3);
            this->update_weights(learning_rate);

            // 4. Lấy giá trị loss về (chỉ tốn cực ít thời gian)
            // Lưu ý: Lệnh này sẽ chặn CPU lại (Sync implicit), nhưng nó nhanh hơn copy cả ảnh nhiều.
            CHECK(cudaMemcpy(&h_loss_val, d_loss_val, sizeof(float), cudaMemcpyDeviceToHost));

            // Loss MSE = Sum / N
            epoch_loss += h_loss_val / (float)current_batch_size;

            // In tiến độ
            float progress = 100.0f * (float)(b + 1) / (float)num_batches;
            std::printf("\r[Optim] Epoch %d/%d - batch %d/%d (%.1f%%)", epoch, n_epoch, b + 1, num_batches, progress);
        }
        std::printf("\n[Optim][Epoch %d] loss = %.6f\n", epoch, epoch_loss / num_batches);
    }
}

float Gpu_Autoencoder_Opt::eval(const Dataset &dataset)
{
    int N = dataset.n;
    int B = this->batch_size;
    double total_loss = 0.0;
    int count = 0;
    for (int b = 0; b < (N + B - 1) / B; ++b)
    {
        int start = b * B;
        int end = (start + B <= N) ? (start + B) : N;
        const float *x = dataset.get_data() + start * 32 * 32 * 3;
        this->forward(x, end - start, 32, 32, 3);
        total_loss += cpu_mse_loss(const_cast<float *>(x), recon_host, end - start, 32, 32, 3);
        count++;
    }
    return (float)(total_loss / count);
}

Dataset Gpu_Autoencoder_Opt::encode(const Dataset &dataset) const
{
    int N = dataset.n;
    int B = this->batch_size;
    int feat_w = 8, feat_h = 8, feat_c = 128;
    Dataset features(N, feat_w, feat_h, feat_c);

    Gpu_Autoencoder_Opt *self = const_cast<Gpu_Autoencoder_Opt *>(this);

    for (int b = 0; b < (N + B - 1) / B; ++b)
    {
        int start = b * B;
        int end = (start + B <= N) ? (start + B) : N;
        int bn = end - start;

        const float *x = dataset.get_data() + start * 32 * 32 * 3;
        self->forward(x, bn, 32, 32, 3);

        float *dst = features.get_data() + start * feat_w * feat_h * feat_c;
        CHECK(cudaMemcpy(dst, d_encoded, bn * feat_w * feat_h * feat_c * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return features;
}

void Gpu_Autoencoder_Opt::save_features(const Dataset &features, const char *filename) const
{
    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s for writing\n", filename);
        return;
    }
    // Write header: num_samples (int), feature_dim (int)
    int n = features.n;
    int dim = features.width * features.height * features.depth;
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&dim, sizeof(int), 1, f);

    // Write data and labels
    fwrite(features.get_data(), sizeof(float), (size_t)n * dim, f);
    fwrite(features.get_labels(), sizeof(int), (size_t)n, f);

    fclose(f);
    printf("Successfully saved features to %s (N=%d, Dim=%d)\n", filename, n, dim);
}

void Gpu_Autoencoder_Opt::save(const char *path) const { printf("Saved to %s\n", path); }
void Gpu_Autoencoder_Opt::load(const char *path) { printf("Loaded from %s\n", path); }
