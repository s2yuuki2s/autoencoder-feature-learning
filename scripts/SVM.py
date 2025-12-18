import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. HÀM ĐỌC DỮ LIỆU TỪ FILE BINARY (C++ OUTPUT)
# ==========================================
def load_features(filename):
    """
    Đọc features và labels từ file nhị phân được xuất bởi C++.
    Format: [n (int), dim (int), data (n*dim floats), labels (n ints)]
    """
    print(f"-> Loading {filename}...")
    try:
        with open(filename, "rb") as f:
            # Đọc header: n (số lượng mẫu), dim (số chiều)
            header = f.read(8)
            if not header:
                raise ValueError("File rỗng")

            n, dim = struct.unpack("ii", header)
            print(f"   Found N={n}, Dim={dim}")

            # Đọc dữ liệu Features (float32)
            data_bytes = f.read(n * dim * 4)
            X = np.frombuffer(data_bytes, dtype=np.float32).reshape(n, dim)

            # Đọc Labels (int32)
            label_bytes = f.read(n * 4)
            y = np.frombuffer(label_bytes, dtype=np.int32)

            return X, y
    except FileNotFoundError:
        print(
            f"Error: Không tìm thấy file {filename}. Hãy chắc chắn bạn đã chạy Phase 3 xong."
        )
        return None, None


# ==========================================
# 2. MAIN WORKFLOW
# ==========================================
def run_phase_4():
    print("=== PHASE 4: SVM CLASSIFICATION ===")

    # 1. Load Data
    X_train, y_train = load_features("train_features_opt.bin")
    X_test, y_test = load_features("test_features_opt.bin")

    if X_train is None or X_test is None:
        return

    # 2. Preprocessing (Standardization)
    # SVM rất nhạy cảm với scale dữ liệu, nên chuẩn hóa là bắt buộc
    print("\n[preprocessing] Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. (Optional) PCA Dimensionality Reduction
    # Vector 8192 chiều rất lớn, train SVM RBF sẽ RẤT LÂU (hàng tiếng đồng hồ).
    # PCA giúp giảm chiều xuống (ví dụ 128 hoặc 256) mà vẫn giữ thông tin chính.
    use_pca = True
    if use_pca:
        n_components = 256  # Giữ lại 256 thành phần quan trọng nhất
        print(
            f"\n[PCA] Reducing dimensions from {X_train.shape[1]} to {n_components}..."
        )
        pca = PCA(n_components=n_components, whiten=True, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"   Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.2f}")

    # 4. Train SVM
    # Cấu hình theo đề bài: Kernel=RBF, C=10 [cite: 158]
    print("\n[SVM] Training SVM classifier (Kernel=RBF, C=10)...")
    print("   (This might take a few minutes depending on data size...)")

    # Lưu ý: Nếu vẫn quá chậm, bạn có thể đổi sang LinearSVC (nhanh hơn nhưng accuracy thấp hơn RBF)
    # clf = LinearSVC(C=10, dual=False, max_iter=1000)

    clf = SVC(
        kernel="rbf", C=10, gamma="scale", cache_size=1000, class_weight="balanced"
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"   Training completed in {train_time:.2f} seconds.")

    # 5. Evaluate
    print("\n[Evaluation] Predicting on Test Set...")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    print("------------------------------------------------")
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    print(f"Inference Time     : {inference_time:.2f} s")
    print("------------------------------------------------")

    # Target theo đề bài là 60-65%
    if acc >= 0.60:
        print("SUCCESS: Đạt mục tiêu (>60%)")
    else:
        print(
            "NOTE: Chưa đạt mục tiêu 60%. Cần train Autoencoder lâu hơn hoặc chỉnh tham số."
        )

    # 6. Classification Report
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    # 7. Confusion Matrix Visualization
    print("[Visualization] Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", values_format="d")
    plt.title(f"Confusion Matrix (Acc: {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Lưu ảnh để báo cáo
    plt.show()


if __name__ == "__main__":
    run_phase_4()
