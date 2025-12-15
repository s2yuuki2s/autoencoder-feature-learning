#include "../../include/cpu/cpu_layers.h"

#define GET_1D_INDEX(i, j, k, width, depth) \
  ((k) + (depth) * ((j) + (i) * (width)))

static inline float my_max(float a, float b) { return a > b ? a : b; }

void cpu_conv2D(float *in, float *filter, float *out,
                int n, int width, int height, int depth, int n_filter)
{
  const int KH = 3, KW = 3;
  for (int img = 0; img < n; ++img)
  {
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        // out shape: (n, height, width, n_filter)
        float *out_pix = out + img * width * height * n_filter + (i * width + j) * n_filter;

        for (int f = 0; f < n_filter; ++f)
        {
          float sum = 0.0f;

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

              // in shape: (n, height, width, depth)
              float *in_pix = in + img * width * height * depth + (row * width + col) * depth;
              // filter shape: (n_filter, KH, KW, depth)
              float *w = filter + f * KH * KW * depth + (fi * KW + fj) * depth;

              for (int c = 0; c < depth; ++c)
              {
                sum += in_pix[c] * w[c];
              }
            }
          }
          out_pix[f] = sum;
        }
      }
    }
  }
}

void cpu_add_bias(float *in, float *bias, float *out,
                  int n, int width, int height, int depth)
{
  // in/out shape: (n, height, width, depth)
  int spatial = width * height;
  for (int img = 0; img < n; ++img)
  {
    for (int idx = 0; idx < spatial; ++idx)
    {
      float *in_pix = in + img * spatial * depth + idx * depth;
      float *out_pix = out + img * spatial * depth + idx * depth;
      for (int c = 0; c < depth; ++c)
      {
        out_pix[c] = in_pix[c] + bias[c];
      }
    }
  }
}

void cpu_relu(float *in, float *out,
              int n, int width, int height, int depth)
{
  int total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
  {
    out[i] = my_max(0.0f, in[i]);
  }
}

void cpu_max_pooling(float *in, float *out,
                     int n, int width, int height, int depth)
{
  // MaxPool 2x2, stride 2
  int out_w = width / 2;
  int out_h = height / 2;

  for (int img = 0; img < n; ++img)
  {
    for (int i = 0; i < out_h; ++i)
    {
      for (int j = 0; j < out_w; ++j)
      {
        // out shape: (n, out_h, out_w, depth)
        float *out_pix = out + img * out_w * out_h * depth + (i * out_w + j) * depth;

        int in_i0 = 2 * i;
        int in_j0 = 2 * j;

        // four neighbors in input
        float *p00 = in + img * width * height * depth + (in_i0 * width + in_j0) * depth;
        float *p01 = in + img * width * height * depth + (in_i0 * width + (in_j0 + 1)) * depth;
        float *p10 = in + img * width * height * depth + ((in_i0 + 1) * width + in_j0) * depth;
        float *p11 = in + img * width * height * depth + ((in_i0 + 1) * width + (in_j0 + 1)) * depth;

        for (int c = 0; c < depth; ++c)
        {
          float v00 = p00[c];
          float v01 = p01[c];
          float v10 = p10[c];
          float v11 = p11[c];
          float m0 = my_max(v00, v01);
          float m1 = my_max(v10, v11);
          out_pix[c] = my_max(m0, m1);
        }
      }
    }
  }
}

void cpu_upsampling(float *in, float *out,
                    int n, int width, int height, int depth)
{
  // Nearest neighbor upsampling by factor 2
  int out_w = width * 2;
  int out_h = height * 2;

  for (int img = 0; img < n; ++img)
  {
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        float *in_pix = in + img * width * height * depth + (i * width + j) * depth;

        // replicate to 4 positions
        for (int di = 0; di < 2; ++di)
        {
          for (int dj = 0; dj < 2; ++dj)
          {
            int oi = 2 * i + di;
            int oj = 2 * j + dj;
            float *out_pix = out + img * out_w * out_h * depth + (oi * out_w + oj) * depth;
            for (int c = 0; c < depth; ++c)
            {
              out_pix[c] = in_pix[c];
            }
          }
        }
      }
    }
  }
}

float cpu_mse_loss(float *expected, float *actual,
                   int n, int width, int height, int depth)
{
  int total = n * width * height * depth;
  double sum = 0.0;
  for (int i = 0; i < total; ++i)
  {
    double diff = (double)actual[i] - (double)expected[i];
    sum += diff * diff;
  }
  return (float)(sum / (double)total);
}
