# QDGrasp

QDGrasp là thư viện AGPL-3.0 cho nghiên cứu và phát triển dexterous grasping
cross-embodiment. Dự án được thiết kế để dùng như package ngay từ đầu; notebook
chỉ là client của public API, không chứa bản sao source.

Python được hỗ trợ từ 3.11 trở lên. Lock tham chiếu dùng Python 3.11 để tái lập;
đây không phải upper bound của thư viện.

## Cài development

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip sync --torch-backend cpu environments/requirements/dev-cpu.lock
uv pip install --no-deps -e .
python -c "import qdgrasp; print(qdgrasp.environment_info())"
```

Hoặc cài exact public commit sau khi đã sync đúng environment lock:

```bash
uv pip install --no-deps "git+https://github.com/ninicom/qdgrasp.git@<commit>"
qdgrasp env
```

CUDA dùng environment riêng và không fallback CPU:

```bash
uv venv --python 3.11 .venv-cuda
source .venv-cuda/bin/activate
uv pip sync --torch-backend cu128 environments/requirements/cu128.lock
uv pip install --no-deps -e .
qdgrasp env --require-cuda
```

Các model/data/checkpoint của DGN2 không được sử dụng. Source tham chiếu nằm
ngoài import path và tuân theo manifest/provenance riêng.

RH56E2 không thuộc robot corpus. Corpus Phase 0 chỉ khóa LEAP, Allegro và Shadow
Hand; asset robot không được đóng gói trong wheel.

Mọi đường dẫn project trong config/manifest/example là tương đối với project
root hoặc working directory. Không commit đường dẫn tuyệt đối của máy phát triển;
đường dẫn hệ điều hành đặc biệt chỉ được dùng khi đọc runtime platform metadata.
