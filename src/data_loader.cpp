#include "data_loader.h"
#include "constants.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ===== Dataset implementation =====

Dataset::Dataset(int n_, int w, int h, int d)
    : data(std::make_unique<float[]>(n_ * w * h * d)),
      labels(std::make_unique<int[]>(n_)),
      n(n_), width(w), height(h), depth(d) {}

Dataset::Dataset(std::unique_ptr<float[]> &&data_,
                 std::unique_ptr<int[]> &&labels_,
                 int n_, int w, int h, int d)
    : data(std::move(data_)),
      labels(std::move(labels_)),
      n(n_), width(w), height(h), depth(d) {}

Dataset::Dataset(const Dataset &other)
    : data(std::make_unique<float[]>(other.n * other.width * other.height * other.depth)),
      labels(std::make_unique<int[]>(other.n)),
      n(other.n), width(other.width), height(other.height), depth(other.depth)
{
  std::memcpy(data.get(), other.data.get(),
              sizeof(float) * n * width * height * depth);
  std::memcpy(labels.get(), other.labels.get(),
              sizeof(int) * n);
}

Dataset &Dataset::operator=(const Dataset &other)
{
  if (this == &other)
    return *this;

  n = other.n;
  width = other.width;
  height = other.height;
  depth = other.depth;

  data = std::make_unique<float[]>(n * width * height * depth);
  labels = std::make_unique<int[]>(n);

  std::memcpy(data.get(), other.data.get(),
              sizeof(float) * n * width * height * depth);
  std::memcpy(labels.get(), other.labels.get(),
              sizeof(int) * n);

  return *this;
}

float *Dataset::get_data() const { return data.get(); }
int *Dataset::get_labels() const { return labels.get(); }

// ===== Helper: read one CIFAR-10 batch =====
//
// mỗi record: 1 byte label + 3072 bytes image (32*32*3)
//

static void read_cifar10_batch(const char *filepath,
                               float *images, int *labels,
                               int offset, int num_samples)
{
  FILE *f = std::fopen(filepath, "rb");
  if (!f)
  {
    std::perror("Failed to open CIFAR-10 file");
    std::fprintf(stderr, "  Path: %s\n", filepath);
    std::exit(EXIT_FAILURE);
  }

  const int record_size = 1 + IMAGE_SIZE;
  size_t buffer_size = (size_t)record_size * num_samples;
  unsigned char *buffer =
      (unsigned char *)std::malloc(buffer_size);
  if (!buffer)
  {
    std::fprintf(stderr, "Failed to allocate buffer for CIFAR-10\n");
    std::exit(EXIT_FAILURE);
  }

  size_t read_bytes = std::fread(buffer, 1, buffer_size, f);
  if (read_bytes != buffer_size)
  {
    std::fprintf(stderr, "Unexpected EOF when reading %s\n", filepath);
    std::exit(EXIT_FAILURE);
  }
  std::fclose(f);

  for (int i = 0; i < num_samples; ++i)
  {
    int dst = offset + i;
    unsigned char *rec = buffer + i * record_size;

    labels[dst] = (int)rec[0];

    for (int j = 0; j < IMAGE_SIZE; ++j)
    {
      unsigned char px = rec[1 + j];
      images[dst * IMAGE_SIZE + j] = px / 255.0f;
    }
  }

  std::free(buffer);
}

// ===== Public API: load_dataset / shuffle / minibatch =====

Dataset load_dataset(const char *dataset_dir, bool is_train)
{
  int num_samples = is_train ? NUM_TRAIN_SAMPLES : NUM_TEST_SAMPLES;

  auto images = std::make_unique<float[]>(num_samples * IMAGE_SIZE);
  auto labels = std::make_unique<int[]>(num_samples);

  if (is_train)
  {
    const int samples_per_batch = NUM_TRAIN_SAMPLES / NUM_BATCHES; // 10000
    std::printf("Loading CIFAR-10 TRAIN from %s\n", dataset_dir);
    for (int b = 1; b <= NUM_BATCHES; ++b)
    {
      char filepath[512];
      std::snprintf(filepath, sizeof(filepath),
                    "%s/data_batch_%d.bin", dataset_dir, b);
      int offset = (b - 1) * samples_per_batch;
      read_cifar10_batch(filepath,
                         images.get(), labels.get(),
                         offset, samples_per_batch);
      std::printf("  Loaded batch %d/5\n", b);
    }
  }
  else
  {
    std::printf("Loading CIFAR-10 TEST from %s\n", dataset_dir);
    char filepath[512];
    std::snprintf(filepath, sizeof(filepath),
                  "%s/test_batch.bin", dataset_dir);
    read_cifar10_batch(filepath,
                       images.get(), labels.get(),
                       0, NUM_TEST_SAMPLES);
  }

  // Quick sanity check range
  float min_val = 1.0f, max_val = 0.0f;
  for (int i = 0; i < num_samples * IMAGE_SIZE; ++i)
  {
    float v = images[i];
    if (v < min_val)
      min_val = v;
    if (v > max_val)
      max_val = v;
  }
  std::printf("  Data range after normalization: [%.3f, %.3f]\n",
              min_val, max_val);

  // Dùng constructor move-ownership
  return Dataset(std::move(images), std::move(labels),
                 num_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
}

void shuffle_dataset(Dataset &dataset)
{
  int n = dataset.n;
  int image_size = dataset.width * dataset.height * dataset.depth;

  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i)
    idx[i] = i;

  std::mt19937 rng(42);
  std::shuffle(idx.begin(), idx.end(), rng);

  auto new_data = std::make_unique<float[]>(n * image_size);
  auto new_labels = std::make_unique<int[]>(n);

  float *src_data = dataset.get_data();
  int *src_lbl = dataset.get_labels();

  for (int i = 0; i < n; ++i)
  {
    int src = idx[i];
    std::memcpy(new_data.get() + i * image_size,
                src_data + src * image_size,
                sizeof(float) * image_size);
    new_labels[i] = src_lbl[src];
  }

  dataset.data = std::move(new_data);
  dataset.labels = std::move(new_labels);
}

std::vector<Dataset> create_minibatches(const Dataset &dataset, int batch_size)
{
  std::vector<Dataset> batches;
  int n = dataset.n;
  int width = dataset.width;
  int height = dataset.height;
  int depth = dataset.depth;
  int image_size = width * height * depth;

  const float *data = dataset.get_data();
  const int *labels = dataset.get_labels();

  int n_batches = (n + batch_size - 1) / batch_size;
  batches.reserve(n_batches);

  for (int b = 0; b < n_batches; ++b)
  {
    int start = b * batch_size;
    int end = std::min(start + batch_size, n);
    int cur_n = end - start;

    Dataset batch(cur_n, width, height, depth);

    std::memcpy(batch.get_data(),
                data + start * image_size,
                sizeof(float) * cur_n * image_size);
    std::memcpy(batch.get_labels(),
                labels + start,
                sizeof(int) * cur_n);

    batches.push_back(std::move(batch));
  }

  return batches;
}

Dataset take_first_n(const Dataset &src, int N_take)
{
  int N = (N_take < src.n) ? N_take : src.n;

  // Tạo Dataset mới với kích thước N
  Dataset dst(N, src.width, src.height, src.depth);

  int image_size = src.width * src.height * src.depth;

  // Copy data
  std::memcpy(dst.get_data(), src.get_data(),
              sizeof(float) * N * image_size);

  // Copy labels
  std::memcpy(dst.get_labels(), src.get_labels(),
              sizeof(int) * N);

  return dst;
}
