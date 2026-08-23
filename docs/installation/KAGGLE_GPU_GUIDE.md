# Hướng dẫn chạy QDGrasp trên Kaggle GPU (T4 / P100)

Tài liệu này hướng dẫn cách đưa toàn bộ thư viện `QDGrasp` lên Kaggle để tận dụng GPU miễn phí, giảm tải hoàn toàn cho CPU và RAM của máy local.

---

## 1. Chuẩn bị mã nguồn

Bạn có thể đưa mã nguồn lên Kaggle theo một trong 2 cách:

### Cách 1: Tạo Kaggle Dataset từ thư mục dự án (Khuyên dùng)
1. Trên máy local, nén toàn bộ thư mục dự án (loại trừ `.venv` và `.git` để file nhẹ):
   ```bash
   tar --exclude='.venv' --exclude='.git' -czvf qdgrasp_source.tar.gz .
   ```
2. Mở Kaggle -> **Datasets** -> **New Dataset** -> Upload file `qdgrasp_source.tar.gz` với tên `qdgrasp-source`.

### Cách 2: Clone từ GitHub (nếu repo là public hoặc dùng Personal Access Token)
```bash
!git clone https://github.com/ninicom/qdgrasp.git
%cd qdgrasp
!git checkout feature/phase3-data-layer
```

---

## 2. Thiết lập Kaggle Notebook

1. Tạo một **New Notebook** trên Kaggle.
2. Trong menu bên phải (**Notebook options**):
   - **Accelerator**: Chọn **GPU T4 x2** hoặc **GPU P100**.
   - **Internet**: Bật **Internet on**.

---

## 3. Các bước chạy trong Kaggle Notebook

### Bước 1: Giải nén mã nguồn & Cài đặt thư viện
```python
import os
import shutil

# Nếu dùng Kaggle Dataset:
if os.path.exists('/kaggle/input/qdgrasp-source/qdgrasp_source.tar.gz'):
    !mkdir -p /kaggle/working/qdgrasp
    !tar -xzf /kaggle/input/qdgrasp-source/qdgrasp_source.tar.gz -C /kaggle/working/qdgrasp
    %cd /kaggle/working/qdgrasp

# Cài đặt các thư viện cần thiết
!pip install -q mujoco>=3.3.0 trimesh>=4.0.0 lightning>=2.6.0 safetensors typer rich pytest pytest-cov
```

### Bước 2: Kiểm tra GPU & Môi trường
```python
import torch

print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name    : {torch.cuda.get_device_name(0)}")
    print(f"Device Memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Bước 3: Chạy kịch bản tự động toàn diện (Entrypoint)
Bạn có thể chạy toàn bộ bài test 310+ test cases, cổng kiểm định Phase 3 và train thử nghiệm trên GPU chỉ với 1 lệnh:
```python
!python scripts/kaggle_entrypoint.py
```

### Bước 4: Chạy Training mô hình trực tiếp bằng Python API
```python
from qdgrasp.api import QDGrasp
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml", seed=42)

result = grasper.train(
    "configs/data/dgn_open_tiny.yaml",
    device=device,
    max_steps=100,
    batch_size=16,
    learning_rate=1e-3,
    run_name="kaggle_gpu_train",
    project_dir="runs/kaggle",
)

print("Training Metrics:", result.metrics)
```

---

## 4. Lợi ích khi chạy trên Kaggle GPU
- **Tiết kiệm tài nguyên máy local**: Không lo bị treo CPU hay tràn RAM khi chạy các batch mô phỏng và training lớn.
- **Tốc độ vượt trội**: Huấn luyện với CUDA PyTorch và Mixed Precision (`amp=True`) nhanh gấp 10-20 lần so với CPU đơn luồng.
- **Lưu trữ kết quả**: Các artifact/checkpoints được lưu trong `/kaggle/working/runs/` có thể tải về máy bất cứ lúc nào qua nút **Download** của Kaggle.
