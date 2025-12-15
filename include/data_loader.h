#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <memory>
#include <vector>

struct Dataset
{
  std::unique_ptr<float[]> data;
  std::unique_ptr<int[]> labels;
  int n;
  int width, height, depth;

  // Constructor cơ bản: tự cấp phát bộ nhớ
  Dataset(int n_, int w, int h, int d);

  // Constructor nhận luôn ownership từ unique_ptr (move)
  Dataset(std::unique_ptr<float[]> &&data_,
          std::unique_ptr<int[]> &&labels_,
          int n_, int w, int h, int d);

  // Copy constructor (deep copy)
  Dataset(const Dataset &other);

  // Copy assignment (deep copy)
  Dataset &operator=(const Dataset &other);

  // Move constructor / move assignment dùng mặc định
  Dataset(Dataset &&) = default;
  Dataset &operator=(Dataset &&) = default;

  float *get_data() const;
  int *get_labels() const;
};

// API Phase 1.1
Dataset load_dataset(const char *dataset_dir, bool is_train);
void shuffle_dataset(Dataset &dataset);
std::vector<Dataset> create_minibatches(const Dataset &dataset, int batch_size);

#endif
