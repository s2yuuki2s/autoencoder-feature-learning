#include "../../include/cpu/cpu_autoencoder.h"
#include "../../include/cpu/cpu_layers.h"
#include "../../include/constants.h"
#include "../../include/data_loader.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ====== Helper: simple random init ======

static float rand_uniform(std::mt19937 &rng, float a, float b) {
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

// ====== Helper: conv2D backward (very naive) ======
//
// in:        (n, h, w, in_c)
// grad_out:  (n, h, w, out_c)
// W:         (out_c, 3, 3, in_c)
// grad_in:   same shape as in (must be zeroed before call)
// grad_W:    same shape as W   (must be zeroed before first batch)
// grad_b:    (out_c)           (must be zeroed before first batch)
//
static void conv2d_backward(
    const float *in,
    const float *grad_out,
    const float *W,
    int n, int width, int height, int in_c, int out_c,
    float *grad_in,
    float *grad_W,
    float *grad_b
) {
    const int KH = 3, KW = 3;

    for (int img = 0; img < n; ++img) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                const float *grad_out_pix = grad_out + img * width * height * out_c
                                          + (i * width + j) * out_c;

                for (int f = 0; f < out_c; ++f) {
                    float go = grad_out_pix[f];
                    grad_b[f] += go;

                    for (int fi = 0; fi < KH; ++fi) {
                        int row = i + fi - KH / 2;
                        if (row < 0 || row >= height) continue;

                        for (int fj = 0; fj < KW; ++fj) {
                            int col = j + fj - KW / 2;
                            if (col < 0 || col >= width) continue;

                            const float *in_pix = in + img * width * height * in_c
                                                + (row * width + col) * in_c;
                            const float *w_ptr  = W + f * KH * KW * in_c
                                                  + (fi * KW + fj) * in_c;
                            float *grad_in_pix   = grad_in + img * width * height * in_c
                                                  + (row * width + col) * in_c;
                            float *grad_w_ptr    = grad_W + f * KH * KW * in_c
                                                  + (fi * KW + fj) * in_c;

                            for (int c = 0; c < in_c; ++c) {
                                grad_w_ptr[c] += go * in_pix[c];
                                grad_in_pix[c] += go * w_ptr[c];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ====== Helper: ReLU backward ======

static void relu_backward(const float *in,
                          const float *grad_out,
                          int total,
                          float *grad_in) {
    for (int i = 0; i < total; ++i) {
        grad_in[i] = (in[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

// ====== Helper: MaxPool 2x2 backward ======
//
// in:       (n, H, W, C)
// out:      (n, H/2, W/2, C)
// grad_out: (n, H/2, W/2, C)
// grad_in:  (n, H, W, C) zeroed before call
//
static void maxpool2x2_backward(const float *in,
                                const float *out,
                                const float *grad_out,
                                int n, int width, int height, int depth,
                                float *grad_in) {
    int out_w = width / 2;
    int out_h = height / 2;

    for (int img = 0; img < n; ++img) {
        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                const float *out_pix      = out + img * out_w * out_h * depth
                                          + (i * out_w + j) * depth;
                const float *grad_out_pix = grad_out + img * out_w * out_h * depth
                                          + (i * out_w + j) * depth;

                int in_i0 = 2 * i;
                int in_j0 = 2 * j;

                const float *p00 = in + img * width * height * depth
                                 + (in_i0 * width + in_j0) * depth;
                const float *p01 = in + img * width * height * depth
                                 + (in_i0 * width + (in_j0 + 1)) * depth;
                const float *p10 = in + img * width * height * depth
                                 + ((in_i0 + 1) * width + in_j0) * depth;
                const float *p11 = in + img * width * height * depth
                                 + ((in_i0 + 1) * width + (in_j0 + 1)) * depth;

                float *g00 = grad_in + img * width * height * depth
                           + (in_i0 * width + in_j0) * depth;
                float *g01 = grad_in + img * width * height * depth
                           + (in_i0 * width + (in_j0 + 1)) * depth;
                float *g10 = grad_in + img * width * height * depth
                           + ((in_i0 + 1) * width + in_j0) * depth;
                float *g11 = grad_in + img * width * height * depth
                           + ((in_i0 + 1) * width + (in_j0 + 1)) * depth;

                for (int c = 0; c < depth; ++c) {
                    float out_val = out_pix[c];
                    float go      = grad_out_pix[c];

                    // route gradient to the max location
                    if (p00[c] == out_val)      g00[c] += go;
                    else if (p01[c] == out_val) g01[c] += go;
                    else if (p10[c] == out_val) g10[c] += go;
                    else                        g11[c] += go;
                }
            }
        }
    }
}

// ====== Helper: Upsampling 2x backward ======
//
// grad_out: (n, 2H, 2W, C)
// grad_in:  (n, H, W, C) zeroed before call
//
static void upsample2x_backward(const float *grad_out,
                                int n, int width, int height, int depth,
                                float *grad_in) {
    int out_w = width * 2;
    int out_h = height * 2;

    for (int img = 0; img < n; ++img) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float *gin = grad_in + img * width * height * depth
                           + (i * width + j) * depth;

                // four children
                const float *g00 = grad_out + img * out_w * out_h * depth
                                 + ((2 * i) * out_w + (2 * j)) * depth;
                const float *g01 = grad_out + img * out_w * out_h * depth
                                 + ((2 * i) * out_w + (2 * j + 1)) * depth;
                const float *g10 = grad_out + img * out_w * out_h * depth
                                 + ((2 * i + 1) * out_w + (2 * j)) * depth;
                const float *g11 = grad_out + img * out_w * out_h * depth
                                 + ((2 * i + 1) * out_w + (2 * j + 1)) * depth;

                for (int c = 0; c < depth; ++c) {
                    gin[c] += g00[c] + g01[c] + g10[c] + g11[c];
                }
            }
        }
    }
}

// ====== Cpu_Autoencoder implementation ======

Cpu_Autoencoder::Cpu_Autoencoder(int batch_size_)
    : W1(nullptr), b1(nullptr),
      W2(nullptr), b2(nullptr),
      W3(nullptr), b3(nullptr),
      W4(nullptr), b4(nullptr),
      W5(nullptr), b5(nullptr),
      h1(nullptr), p1(nullptr), h2(nullptr), encoded(nullptr),
      h3(nullptr), u1(nullptr), h4(nullptr), u2(nullptr), recon(nullptr),
      dW1(nullptr), db1(nullptr),
      dW2(nullptr), db2(nullptr),
      dW3(nullptr), db3(nullptr),
      dW4(nullptr), db4(nullptr),
      dW5(nullptr), db5(nullptr),
      batch_size(batch_size_)
{
    int B = batch_size;

    // weight shapes (out_c, 3, 3, in_c)
    int W1_size = 256 * 3 * 3 * 3;
    int W2_size = 128 * 3 * 3 * 256;
    int W3_size = 128 * 3 * 3 * 128;
    int W4_size = 256 * 3 * 3 * 128;
    int W5_size = 3   * 3 * 3 * 256;

    W1 = new float[W1_size]; b1 = new float[256];
    W2 = new float[W2_size]; b2 = new float[128];
    W3 = new float[W3_size]; b3 = new float[128];
    W4 = new float[W4_size]; b4 = new float[256];
    W5 = new float[W5_size]; b5 = new float[3];

    dW1 = new float[W1_size]; db1 = new float[256];
    dW2 = new float[W2_size]; db2 = new float[128];
    dW3 = new float[W3_size]; db3 = new float[128];
    dW4 = new float[W4_size]; db4 = new float[256];
    dW5 = new float[W5_size]; db5 = new float[3];

    // activations (max batch = B)
    h1      = new float[B * 32 * 32 * 256];
    p1      = new float[B * 16 * 16 * 256];
    h2      = new float[B * 16 * 16 * 128];
    encoded = new float[B * 8  * 8  * 128];
    h3      = new float[B * 8  * 8  * 128];
    u1      = new float[B * 16 * 16 * 128];
    h4      = new float[B * 16 * 16 * 256];
    u2      = new float[B * 32 * 32 * 256];
    recon   = new float[B * 32 * 32 * 3];
}

Cpu_Autoencoder::~Cpu_Autoencoder() {
    delete[] W1; delete[] b1;
    delete[] W2; delete[] b2;
    delete[] W3; delete[] b3;
    delete[] W4; delete[] b4;
    delete[] W5; delete[] b5;

    delete[] dW1; delete[] db1;
    delete[] dW2; delete[] db2;
    delete[] dW3; delete[] db3;
    delete[] dW4; delete[] db4;
    delete[] dW5; delete[] db5;

    delete[] h1;
    delete[] p1;
    delete[] h2;
    delete[] encoded;
    delete[] h3;
    delete[] u1;
    delete[] h4;
    delete[] u2;
    delete[] recon;
}

void Cpu_Autoencoder::init_weights(int seed) {
    std::mt19937 rng(seed);
    auto init_array = [&](float *w, int size, float scale) {
        for (int i = 0; i < size; ++i)
            w[i] = rand_uniform(rng, -scale, scale);
    };

    init_array(W1, 256 * 3 * 3 * 3,     0.05f);
    init_array(W2, 128 * 3 * 3 * 256,   0.05f);
    init_array(W3, 128 * 3 * 3 * 128,   0.05f);
    init_array(W4, 256 * 3 * 3 * 128,   0.05f);
    init_array(W5, 3   * 3 * 3 * 256,   0.05f);

    std::memset(b1, 0, 256 * sizeof(float));
    std::memset(b2, 0, 128 * sizeof(float));
    std::memset(b3, 0, 128 * sizeof(float));
    std::memset(b4, 0, 256 * sizeof(float));
    std::memset(b5, 0, 3   * sizeof(float));
}

// Full forward for a batch (n <= batch_size)
void Cpu_Autoencoder::forward(const float* input,
                              int n, int width, int height, int depth)
{
    (void)width; (void)height; (void)depth; // assume 32x32x3

    // Encoder
    // conv1: in (n,32,32,3) -> h1 (n,32,32,256)
    cpu_conv2D(const_cast<float*>(input), W1, h1,
               n, 32, 32, 3, 256);
    cpu_add_bias(h1, b1, h1, n, 32, 32, 256);
    cpu_relu(h1, h1,       n, 32, 32, 256);

    // pool1: h1 -> p1 (n,16,16,256)
    cpu_max_pooling(h1, p1, n, 32, 32, 256);

    // conv2: p1 (n,16,16,256) -> h2 (n,16,16,128)
    cpu_conv2D(p1, W2, h2,
               n, 16, 16, 256, 128);
    cpu_add_bias(h2, b2, h2, n, 16, 16, 128);
    cpu_relu(h2, h2,       n, 16, 16, 128);

    // pool2: h2 -> encoded (n,8,8,128)
    cpu_max_pooling(h2, encoded, n, 16, 16, 128);

    // conv3: encoded (n,8,8,128) -> h3 (n,8,8,128)
    cpu_conv2D(encoded, W3, h3,
               n, 8, 8, 128, 128);
    cpu_add_bias(h3, b3, h3, n, 8, 8, 128);
    cpu_relu(h3, h3,       n, 8, 8, 128);

    // upsample1: h3 (n,8,8,128) -> u1 (n,16,16,128)
    cpu_upsampling(h3, u1, n, 8, 8, 128);

    // conv4: u1 (n,16,16,128) -> h4 (n,16,16,256)
    cpu_conv2D(u1, W4, h4,
               n, 16, 16, 128, 256);
    cpu_add_bias(h4, b4, h4, n, 16, 16, 256);
    cpu_relu(h4, h4,       n, 16, 16, 256);

    // upsample2: h4 (n,16,16,256) -> u2 (n,32,32,256)
    cpu_upsampling(h4, u2, n, 16, 16, 256);

    // conv5: u2 (n,32,32,256) -> recon (n,32,32,3)
    cpu_conv2D(u2, W5, recon,
               n, 32, 32, 256, 3);
    cpu_add_bias(recon, b5, recon, n, 32, 32, 3);
}

// Backward pass (naive, but works) for autoencoder
void Cpu_Autoencoder::backward(const float* input,
                               const float* target,
                               int n, int width, int height, int depth)
{
    (void)width; (void)height; (void)depth;

    // Zero gradients for weights/biases
    std::memset(dW1, 0, sizeof(float) * 256 * 3 * 3 * 3);
    std::memset(dW2, 0, sizeof(float) * 128 * 3 * 3 * 256);
    std::memset(dW3, 0, sizeof(float) * 128 * 3 * 3 * 128);
    std::memset(dW4, 0, sizeof(float) * 256 * 3 * 3 * 128);
    std::memset(dW5, 0, sizeof(float) * 3   * 3 * 3 * 256);

    std::memset(db1, 0, sizeof(float) * 256);
    std::memset(db2, 0, sizeof(float) * 128);
    std::memset(db3, 0, sizeof(float) * 128);
    std::memset(db4, 0, sizeof(float) * 256);
    std::memset(db5, 0, sizeof(float) * 3);

    // Allocate grad buffers for activations (per batch)
    int B = n;

    float *g_recon   = new float[B * 32 * 32 * 3];
    float *g_u2      = new float[B * 32 * 32 * 256];
    float *g_h4      = new float[B * 16 * 16 * 256];
    float *g_u1      = new float[B * 16 * 16 * 128];
    float *g_h3      = new float[B * 8  * 8  * 128];
    float *g_encoded = new float[B * 8  * 8  * 128];
    float *g_h2      = new float[B * 16 * 16 * 128];
    float *g_p1      = new float[B * 16 * 16 * 256];
    float *g_h1      = new float[B * 32 * 32 * 256];
    float *g_input   = new float[B * 32 * 32 * 3];

    std::memset(g_recon,   0, sizeof(float) * B * 32 * 32 * 3);
    std::memset(g_u2,      0, sizeof(float) * B * 32 * 32 * 256);
    std::memset(g_h4,      0, sizeof(float) * B * 16 * 16 * 256);
    std::memset(g_u1,      0, sizeof(float) * B * 16 * 16 * 128);
    std::memset(g_h3,      0, sizeof(float) * B * 8  * 8  * 128);
    std::memset(g_encoded, 0, sizeof(float) * B * 8  * 8  * 128);
    std::memset(g_h2,      0, sizeof(float) * B * 16 * 16 * 128);
    std::memset(g_p1,      0, sizeof(float) * B * 16 * 16 * 256);
    std::memset(g_h1,      0, sizeof(float) * B * 32 * 32 * 256);
    std::memset(g_input,   0, sizeof(float) * B * 32 * 32 * 3);

    // dL/d(recon) = (2/N)* (recon - target)
    int total = B * 32 * 32 * 3;
    float scale = 2.0f / (float)total;
    for (int i = 0; i < total; ++i) {
        g_recon[i] = scale * (recon[i] - target[i]);
    }

    // Backprop through conv5 + bias: u2 -> recon
    conv2d_backward(
        u2, g_recon, W5,
        B, 32, 32, 256, 3,
        g_u2, dW5, db5
    );

    // Backprop through upsample2: h4 -> u2
    upsample2x_backward(
        g_u2,
        B, 16, 16, 256,
        g_h4
    );

    // Backprop through ReLU4: u1 -> h4
    relu_backward(
        h4,
        g_h4,
        B * 16 * 16 * 256,
        g_h4
    );

    // Backprop through conv4: u1 -> h4
    conv2d_backward(
        u1, g_h4, W4,
        B, 16, 16, 128, 256,
        g_u1, dW4, db4
    );

    // Backprop through upsample1: h3 -> u1
    upsample2x_backward(
        g_u1,
        B, 8, 8, 128,
        g_h3
    );

    // Backprop through ReLU3: encoded -> h3
    relu_backward(
        h3,
        g_h3,
        B * 8 * 8 * 128,
        g_h3
    );

    // Backprop through conv3: encoded -> h3
    conv2d_backward(
        encoded, g_h3, W3,
        B, 8, 8, 128, 128,
        g_encoded, dW3, db3
    );

    // Backprop through pool2: h2 -> encoded
    maxpool2x2_backward(
        h2, encoded, g_encoded,
        B, 16, 16, 128,
        g_h2
    );

    // Backprop through ReLU2: p1 -> h2
    relu_backward(
        h2,
        g_h2,
        B * 16 * 16 * 128,
        g_h2
    );

    // Backprop through conv2: p1 -> h2
    conv2d_backward(
        p1, g_h2, W2,
        B, 16, 16, 256, 128,
        g_p1, dW2, db2
    );

    // Backprop through pool1: h1 -> p1
    maxpool2x2_backward(
        h1, p1, g_p1,
        B, 32, 32, 256,
        g_h1
    );

    // Backprop through ReLU1: input -> h1
    relu_backward(
        h1,
        g_h1,
        B * 32 * 32 * 256,
        g_h1
    );

    // Backprop through conv1: input -> h1
    conv2d_backward(
        input, g_h1, W1,
        B, 32, 32, 3, 256,
        g_input, dW1, db1
    );

    delete[] g_recon;
    delete[] g_u2;
    delete[] g_h4;
    delete[] g_u1;
    delete[] g_h3;
    delete[] g_encoded;
    delete[] g_h2;
    delete[] g_p1;
    delete[] g_h1;
    delete[] g_input;
}

// SGD update
void Cpu_Autoencoder::update_weights(float lr) {
    int W1_size = 256 * 3 * 3 * 3;
    int W2_size = 128 * 3 * 3 * 256;
    int W3_size = 128 * 3 * 3 * 128;
    int W4_size = 256 * 3 * 3 * 128;
    int W5_size = 3   * 3 * 3 * 256;

    for (int i = 0; i < W1_size; ++i) W1[i] -= lr * dW1[i];
    for (int i = 0; i < W2_size; ++i) W2[i] -= lr * dW2[i];
    for (int i = 0; i < W3_size; ++i) W3[i] -= lr * dW3[i];
    for (int i = 0; i < W4_size; ++i) W4[i] -= lr * dW4[i];
    for (int i = 0; i < W5_size; ++i) W5[i] -= lr * dW5[i];

    for (int i = 0; i < 256; ++i) b1[i] -= lr * db1[i];
    for (int i = 0; i < 128; ++i) b2[i] -= lr * db2[i];
    for (int i = 0; i < 128; ++i) b3[i] -= lr * db3[i];
    for (int i = 0; i < 256; ++i) b4[i] -= lr * db4[i];
    for (int i = 0; i < 3;   ++i) b5[i] -= lr * db5[i];
}

// Train on full dataset
void Cpu_Autoencoder::fit(const Dataset& dataset,
                          int n_epoch, int batch_size_,
                          float learning_rate,
                          int seed,
                          int checkpoint,
                          const char* output_dir)
{
    (void)seed;
    (void)checkpoint;
    (void)output_dir;

    int N = dataset.n;
    int image_size = IMAGE_SIZE;

    for (int epoch = 1; epoch <= n_epoch; ++epoch) {
        // shuffle
        Dataset shuffled = dataset;
        shuffle_dataset(shuffled);

        // mini-batches
        std::vector<Dataset> batches =
            create_minibatches(shuffled, batch_size_);

        double epoch_loss = 0.0;
        int num_batches = (int)batches.size();

        for (int b = 0; b < num_batches; ++b) {
          Dataset &batch = batches[b];
          int bn = batch.n;

          const float *x = batch.get_data();

          forward(x, bn, 32, 32, 3);
          float loss = cpu_mse_loss(batch.get_data(), recon,
                              bn, 32, 32, 3);
          epoch_loss += loss;

          backward(batch.get_data(), batch.get_data(),
             bn, 32, 32, 3);
          update_weights(learning_rate);

          // === thêm đoạn này để in % ===
          float progress = 100.0f * (float)(b + 1) / (float)num_batches;
          std::printf("\rEpoch %d/%d - batch %d/%d (%.1f%%)",
                epoch, n_epoch, b + 1, num_batches, progress);
          std::fflush(stdout);
      }
// sau khi xong 1 epoch, in xuống dòng
std::printf("\n");


        epoch_loss /= (double)num_batches;
        std::printf("[Epoch %d] loss = %.6f\n", epoch, epoch_loss);
    }
}

// Evaluate reconstruction MSE on full dataset
float Cpu_Autoencoder::eval(const Dataset& dataset) const {
    int N = dataset.n;
    int image_size = IMAGE_SIZE;

    int batch = this->batch_size;
    int num_batches = (N + batch - 1) / batch;

    double total_loss = 0.0;
    int count = 0;

    // NOTE: const_cast to call non-const forward; this is fine for eval
    Cpu_Autoencoder *self = const_cast<Cpu_Autoencoder*>(this);

    for (int b = 0; b < num_batches; ++b) {
        int start = b * batch;
        int end   = std::min(start + batch, N);
        int bn    = end - start;

        const float *x = dataset.get_data() + start * image_size;

        self->forward(x, bn, 32, 32, 3);
        float loss = cpu_mse_loss(const_cast<float*>(x), self->recon,
                                  bn, 32, 32, 3);
        total_loss += (double)loss;
        ++count;
    }

    return (float)(total_loss / (double)count);
}

// Encode: CURRENTLY stub (return zeros). You can refine later if cần cho Phase 2.4
Dataset Cpu_Autoencoder::encode(const Dataset& dataset) const {
    int N = dataset.n;
    int feat_w = 8, feat_h = 8, feat_c = 128;

    // allocate feature dataset (all zeros)
    Dataset features(N, feat_w, feat_h, feat_c);
    std::memset(features.get_data(), 0,
                sizeof(float) * N * feat_w * feat_h * feat_c);
    return features;
}

// Save/load: simple stub for now
void Cpu_Autoencoder::save(const char* path) const {
    std::printf("[Cpu_Autoencoder::save] Not implemented, path=%s\n", path);
}

void Cpu_Autoencoder::load(const char* path) {
    std::printf("[Cpu_Autoencoder::load] Not implemented, path=%s\n", path);
}
