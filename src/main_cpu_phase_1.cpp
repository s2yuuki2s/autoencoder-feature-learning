#include <cstdio>
#include <chrono>

#include "../include/data_loader.h"
#include "../include/cpu/cpu_autoencoder.h"

int main(int argc, char** argv) {
    const char* dataset_dir =
        (argc > 1 ? argv[1] : "./data/cifar-10-batches-bin");

    int n_epoch    = 20;
    int batch_size = 32;
    float lr       = 1e-3f;
    int seed       = 42;
    int checkpoint = 0;
    const char* out_dir = "./checkpoints_cpu";

    std::printf("=== Phase 1: CPU Baseline on CIFAR-10 ===\n");
    Dataset train_ds = load_dataset(dataset_dir, /*is_train=*/true);
    shuffle_dataset(train_ds);

    Cpu_Autoencoder ae(batch_size);
    ae.init_weights(seed);

    auto t0 = std::chrono::high_resolution_clock::now();
    ae.fit(train_ds, n_epoch, batch_size, lr, seed, checkpoint, out_dir);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_sec =
        std::chrono::duration<double>(t1 - t0).count();
    double time_per_epoch = total_sec / n_epoch;

    std::printf("CPU total training time: %.2f s\n", total_sec);
    std::printf("CPU time per epoch    : %.2f s\n", time_per_epoch);

    float mse = ae.eval(train_ds);
    std::printf("Final reconstruction MSE: %.6f\n", mse);

    ae.save("./checkpoints_cpu/autoencoder_cpu.bin");
    return 0;
}
