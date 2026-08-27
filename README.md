# QDGrasp

QDGrasp là thư viện AGPL-3.0 cho nghiên cứu và phát triển dexterous grasping
cross-embodiment. Dự án được thiết kế để dùng như package ngay từ đầu; notebook
chỉ là client của public API, không chứa bản sao source.

Python được hỗ trợ từ 3.11 trở lên. Lock tham chiếu dùng Python 3.11 để tái lập;
đây không phải upper bound của thư viện.

> **Phạm vi robot hiện tại:** workload, dataset, checkpoint và release gate mới
> chỉ dùng LEAP Hand và Wonik Allegro. Shadow Hand đang tạm dừng vì cấu hình
> underactuated/contact-control còn phức tạp. Preset và evidence Shadow vẫn được
> giữ cho compatibility/audit nhưng không thuộc default workload. Xem
> `docs/decisions/0008-temporary-shadow-hand-pause.md`.

## Cài development

Cài thông thường theo pattern package của YOLO — dependency dùng compatible
version ranges:

```bash
pip install "git+https://github.com/ninicom/qdgrasp.git"
qdgrasp env
```

Để tái lập nghiên cứu/CI chính xác, dùng lock profile:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip sync --torch-backend cpu environments/requirements/dev-cpu.lock
uv pip install --no-deps -e .
python -c "import qdgrasp; print(qdgrasp.environment_info())"
```

Có thể pin exact public commit sau khi đã sync đúng environment lock:

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

## Lifecycle Phase 1

Framework hiện chạy được đủ `train/val/predict/export` trên dummy model/data.
Dummy chỉ là fixture để kiểm lifecycle; model thật thuộc Phase 4.

```bash
qdgrasp train --model qdgrasp-dummy-n.yaml --data dummy-tiny.yaml \
  --robot dummy-hand.yaml --device cpu --max-steps 20 --run-name demo
qdgrasp val --model qdgrasp-dummy-n.yaml --data dummy-tiny.yaml \
  --robot dummy-hand.yaml --weights runs/demo/bundle
qdgrasp export --model qdgrasp-dummy-n.yaml --robot dummy-hand.yaml \
  --format torchscript --out-dir runs/demo/export
```

```python
from qdgrasp import QDGrasp

grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="dummy-hand.yaml")
result = grasper.train(data="dummy-tiny.yaml", device="cpu", max_steps=20)
metrics = grasper.val(data="dummy-tiny.yaml", device="cpu")
grasps = grasper.predict(points, device="cpu")
```

Mỗi run ghi `results.json`, public bundle safetensors và resume state riêng.
CLI chỉ nhận subcommand cùng `--flag`; key lạ trong YAML hoặc trên CLI là lỗi.
CUDA không bao giờ fallback về CPU.

Các model/data/checkpoint của DGN2 không được sử dụng. Source tham chiếu nằm
ngoài import path và tuân theo manifest/provenance riêng.

RH56E2 không thuộc robot corpus. Corpus lịch sử Phase 0 từng khóa LEAP, Allegro
và Shadow Hand; theo ADR-0008, active corpus hiện chỉ gồm LEAP và Allegro, còn
Shadow ở trạng thái paused/experimental. Asset robot không được đóng gói trong
wheel.

Robot presets trong wheel dùng URI `asset://…`. Trước khi dựng `RobotSpec` cho
một preset đó, cung cấp checkout asset đã pin qua
`QDGRASP_ROBOT_ASSETS_ROOT=/path/to/robot-assets`; main wheel không tự tải hoặc
phân phối lại MJCF/mesh robot.

Mọi đường dẫn project trong config/manifest/example là tương đối với project
root hoặc working directory. Không commit đường dẫn tuyệt đối của máy phát triển;
đường dẫn hệ điều hành đặc biệt chỉ được dùng khi đọc runtime platform metadata.
