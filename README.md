# autoencoder-feature-learning
Autoencoder Feature Learning on CIFAR-10

Parallel Programming (Lập trình song song) – Final Project

1. Giới thiệu

Dự án này triển khai một hệ thống autoencoder cho bài toán học đặc trưng không giám sát (unsupervised feature learning) trên bộ dữ liệu CIFAR-10, phục vụ cho đồ án cuối kỳ môn Parallel Programming (Lập trình song song).

Trọng tâm của dự án là xây dựng pipeline tính toán theo từng giai đoạn, bắt đầu từ baseline chạy tuần tự trên CPU và mở rộng sang triển khai song song trên GPU bằng CUDA, qua đó đánh giá tác động của lập trình song song đối với hiệu năng của mô hình học sâu.

2. Mục tiêu dự án

Mục tiêu chính của dự án bao gồm xây dựng mô hình autoencoder cho bài toán học đặc trưng ảnh, triển khai baseline tính toán trên CPU, áp dụng lập trình song song trên GPU để tăng tốc quá trình tính toán, so sánh hiệu năng giữa các phiên bản CPU, GPU cơ bản và GPU tối ưu, đồng thời đánh giá vai trò của song song hóa trong huấn luyện mô hình học sâu.

3. Cấu trúc thư mục

```

autoencoder-feature-learning-main/
├── include/
│ ├── constants.h
│ ├── data_loader.h
│ ├── cpu/
│ │ ├── cpu_layers.h
│ │ └── cpu_autoencoder.h
│ └── gpu/
│ ├── gpu_layers.h
│ ├── gpu_autoencoder.h
│ └── gpu_autoencoder_opt.h
├── src/
│ ├── data_loader.cpp
│ ├── main_cpu_phase_1.cpp
│ ├── main_gpu_phase_2.cu
│ ├── main_gpu_phase_3.cu
│ ├── cpu/
│ │ ├── cpu_layers.cpp
│ │ └── cpu_autoencoder.cpp
│ └── gpu/
│ ├── gpu_layers.cu
│ ├── gpu_autoencoder.cu
│ └── gpu_autoencoder_opt.cu
├── scripts/
│ └── download_cifar10.py
├── CSC14120_2025_Final Project.pdf
└── README.md

```

4. Mô tả các thành phần chính

Thư mục include/ chứa các file header khai báo cấu trúc dữ liệu, lớp và giao diện hàm cho toàn bộ hệ thống. File constants.h định nghĩa các hằng số cấu hình dùng chung, trong khi data_loader.h khai báo giao diện nạp và tiền xử lý dữ liệu CIFAR-10. Nhánh cpu/ chứa các khai báo cho các toán tử và kiến trúc autoencoder chạy tuần tự trên CPU, còn nhánh gpu/ chứa các khai báo kernel CUDA và các phiên bản autoencoder chạy song song trên GPU.

Thư mục src/ chứa phần cài đặt chi tiết cho các module đã được khai báo trong include/. File data_loader.cpp hiện thực logic đọc và xử lý dữ liệu CIFAR-10. Các thư mục con cpu/ và gpu/ lần lượt hiện thực các toán tử và mô hình autoencoder trên CPU và GPU.

Các file main được tách riêng theo từng Phase nhằm phục vụ việc đánh giá hiệu năng. File main_cpu_phase_1.cpp thực hiện CPU baseline. File main_gpu_phase_2.cu triển khai forward pass song song trên GPU. File main_gpu_phase_3.cu thực hiện huấn luyện mô hình autoencoder trên GPU và áp dụng các tối ưu.

Thư mục scripts/ chứa script download_cifar10.py dùng để tải và giải nén bộ dữ liệu CIFAR-10, giúp đảm bảo khả năng tái lập của quá trình thực nghiệm.

5. Các Phase thực hiện

Phase 1 tập trung xây dựng CPU baseline với các phép toán được thực hiện tuần tự trên CPU và đóng vai trò làm chuẩn so sánh. Phase 2 chuyển forward pass của mô hình sang GPU bằng CUDA nhằm khai thác song song dữ liệu và song song tính toán. Phase 3 mở rộng sang huấn luyện mô hình trên GPU và áp dụng các kỹ thuật tối ưu hóa kernel nhằm cải thiện hiệu năng.

6. Cách chạy trên Google Colab

Trước khi chạy, cần kiểm tra GPU và CUDA bằng các lệnh nvidia-smi và nvcc --version. Dữ liệu CIFAR-10 được tải bằng script scripts/download_cifar10.py.

CPU baseline được compile và chạy bằng nvcc với các file trong nhánh cpu/. GPU forward pass và GPU training được compile và chạy bằng nvcc với các file tương ứng trong nhánh gpu/.

7. Kết luận

Dự án minh họa rõ ràng vai trò của lập trình song song trong việc tăng tốc các bài toán học sâu. So sánh giữa CPU baseline và các phiên bản GPU cho thấy song song hóa giúp cải thiện đáng kể hiệu năng, đặc biệt đối với các toán tử có chi phí tính toán cao như convolution. Kết quả này phù hợp với mục tiêu và yêu cầu của môn học Parallel Programming.

8. Tài liệu tham khảo

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Course Staff. (2025). Parallel Programming – Final Project Specification.
