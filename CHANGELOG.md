# Changelog

## [Unreleased]

### Added

- Public façade `QDGrasp` và `GraspResults` với `.cpu()/.to()/.numpy()/.save()/.summary()`.
- Config layer khai báo: schema có version, allowlist registry, requested/effective
  runtime và từ chối unknown/dead key.
- Lightning Fabric runner: deterministic seed, callbacks, EMA, checkpoint bundle
  safetensors, resume state tách riêng và result bundle `results.json`.
- CLI subcommand `train/val/predict/export` cạnh `env`.
- Dummy model/dataset và preset YAML để chạy toàn bộ lifecycle trên CPU.
- Export TorchScript/ONNX kèm metadata sidecar và round-trip verification.
- `scripts/check_phase1.py` (CPU gate) và `scripts/phase1_cuda_smoke.py` (CUDA gate).

### Limitations

- Dummy model không mang bất kỳ tuyên bố nghiên cứu nào.
- Robot layer, data layer và model thật vẫn thuộc Phase 2–4.
- Phần CUDA của gate Phase 1 chưa chạy: máy phát triển hiện chỉ có `torch+cpu`.

## [0.1.0-alpha.1] - 2026-08-21

### Added

- `PLAN-V2` cho kiến trúc thư viện QDGrasp theo YAML và core CPU/CUDA.
- Bộ tài liệu session, metrics, revision, independent review và raw evidence.
- Validator tài liệu cùng bốn regression tests.
- GitFlow `feature/*` → `develop` → `release/*` → `main`.
- Hooks, Conventional Commits và automation start/finish feature/release.

### Limitations

- Đây chỉ là prerelease bootstrap tài liệu và quy trình.
- Chưa clone/pin upstream, chưa xác minh license, chưa có package/model,
  checkpoint, dataset, CPU/GPU runtime, export, simulator hoặc benchmark.
