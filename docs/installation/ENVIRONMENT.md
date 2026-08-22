---
document_id: INSTALL-ENV-001
document_type: protocol
title: Môi trường cài đặt chuẩn cho QDGrasp research
version: 1.2.0
status: active
date: 2026-08-22
revises: INSTALL-ENV-001@1.1.0
necessity: N2
impact: Khóa Python, PyTorch, simulation, geometry và tooling cho CPU/CUDA parity.
revision_reason: Maintainer chốt mọi workload CUDA phải chạy trên GPU NVIDIA thật, không dùng CPU fallback làm bằng chứng.
---

# Môi trường cài đặt chuẩn

## Contract đã chốt

| Thành phần | Phiên bản/profile |
| --- | --- |
| OS chuẩn | Linux x86_64 |
| Python package support | `>=3.11` |
| Python reference lock | 3.11.x |
| PyTorch | 2.11.0 |
| CPU wheel | `torch==2.11.0+cpu` |
| NVIDIA wheel | `torch==2.11.0+cu128` |
| Lightning Fabric | `lightning==2.6.5` |
| MuJoCo | `mujoco==3.12.0` |
| Geometry/numeric | NumPy 2.4.6, SciPy 1.17.1, trimesh 4.12.2 |
| Config | Pydantic 2.13.4 + PyYAML 6.0.3 |
| Checkpoint | safetensors 0.8.0 |

Fingerprint, SHA-256 và trạng thái xác minh từng profile nằm trong
`environments/environment.lock.yaml`.

Profile tái lập Phase 0 được resolve/test bằng Python 3.11, nhưng package không
chặn Python 3.12 trở lên. Mỗi minor mới vẫn phải resolve dependency và chạy test;
không cài dependencies QDGrasp vào system Python của máy phát triển.

## Cài thư viện bằng uv

QDGrasp luôn được cài như package. Lock profile được sync trước để chọn đúng
Torch backend; sau đó cài source/wheel với `--no-deps`. Không chạy trực tiếp bằng
cách thêm repository root vào `PYTHONPATH`.

Package metadata theo pattern Ultralytics/YOLO: Python và dependencies dùng
compatible lower bounds để `pip install` dễ trên môi trường hiện có. Các file
lock bên dưới mới là exact contract cho research, CI và paper evidence; package
range không thay thế lock.

CPU là profile chuẩn cho CI, schema/FK/unit tests, debugging và inference/export
smoke; CPU không phải profile train/benchmark chính thức:

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip sync --torch-backend cpu environments/requirements/cpu.lock
uv pip install --no-deps -e .
qdgrasp env
python scripts/check_environment.py --profile cpu
```

CUDA là profile bắt buộc cho train, AMP, distributed, full evaluation, CUDA
parity và benchmark, cùng PyTorch 2.11 nhưng dùng cu128:

```bash
uv python install 3.11
uv venv --python 3.11 .venv-cuda
source .venv-cuda/bin/activate
uv pip sync --torch-backend cu128 environments/requirements/cu128.lock
uv pip install --no-deps -e .
qdgrasp env --require-cuda
python scripts/check_environment.py --profile cuda
```

Profile CUDA chỉ được đánh dấu pass khi `torch.version.cuda == "12.8"`,
`torch.cuda.is_available()` là true và smoke test model chạy trên GPU. CUDA
toolkit hệ thống không bắt buộc cho wheel runtime; driver NVIDIA tương thích vẫn
bắt buộc.

Không có CPU fallback cho lệnh yêu cầu `device=cuda`. Nếu GPU/driver/VRAM không
đủ, lệnh phải fail trước khi train và phase giữ trạng thái blocked/in progress.
Dry-run dependency chỉ chứng minh lock resolve, không chứng minh CUDA runtime.

## Extras

Môi trường mặc định không cài PyTorch3D, pointops, GraspQP, Isaac Gym, cuRobo,
ONNX hay các dependencies từ repository tham chiếu. Export dùng lock riêng:

```bash
uv pip sync --torch-backend cpu environments/requirements/export-cpu.lock
```

Dev/lint/test dùng:

```bash
uv pip sync --torch-backend cpu environments/requirements/dev-cpu.lock
```

Không trộn CPU và CUDA lock trong cùng virtual environment. Khi đổi direct
dependency, sửa file `.in`, regenerate cả hai backend lock và chạy lại CPU/CUDA
parity trước khi merge.

## Gate tối thiểu

```bash
python scripts/check_environment.py --profile cpu
python3 scripts/check_docs.py --root .
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

CPU reference đã được xác minh cục bộ bằng Python 3.11. CUDA gate Phase 0 đã
được xác minh độc lập trên Kaggle bằng Python 3.12.13, PyTorch 2.11.0+cu128,
CUDA 12.8 và Tesla T4. Run 8 pass CUDA validation, ba AMP train-step,
checkpoint resume và MuJoCo forward trên exact project commit
`e20ce61bb44dd4d0787d1924e34335f91cec5289`.

Notebook nằm ở repository tách biệt
`https://github.com/ninicom/qdgrasp-cuda-kaggle`; evidence công khai nằm tại
`evidence/run-008-pass/phase0_cuda_evidence.json`, SHA-256
`00e75d727393ff7dae8c2c893362a6df34ba5abd265cb346823058af86b69291`.
Notebook cài exact public commit của `https://github.com/ninicom/qdgrasp`, gọi
`qdgrasp.require_cuda()` và không chứa bản sao package source.
