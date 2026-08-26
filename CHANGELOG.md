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
- Phase 2 — robot layer: `RobotSpec` với named joints/limits, HandGraph, URDF/MJCF
  parser và mesh resolver, frame transforms, FK batch và MuJoCo grasp adapter cho
  LEAP, Allegro và Shadow Hand.
- Phase 2 — `robot_assets.lock.yaml`, `scripts/check_robot_assets.py` và
  `scripts/check_phase2.py`; CUDA FK parity chạy bằng `scripts/phase2_cuda_fk_parity.py`.
- Phase 3 — data layer: deterministic RNG contract (`qdgrasp/dataset/rng.py`),
  procedural object generator và asset manifest, collision representation guard,
  analytic point-cloud render, dataset schema/manifest/shards/splits/loader.
- Phase 3 — grasp pipeline có typed contracts và recipe allowlist: proposal
  (`surface_fixed`, `region_opposition`, `wrench_guided`), batched DLS solvers,
  GWS+gravity certifier, contact-load observer và MuJoCo dynamic validator.
- Phase 3.2/3.2.1 — underactuated transmission cho Shadow, exact collision,
  task-space command, dynamic predicate và generated-reachable full-flow evidence
  cho cả ba hand.
- Phase 3.3 — scene layer: adapters (GraspNet-1Billion, DexGraspNet 2.0,
  GraspClutter6D), native scene builders, whole-scene clearance, multi-object
  rollout, camera observations và release `datasets/qdgrasp-scene-tiny/`.
- Gate script cho từng lớp: `scripts/check_phase3.py`, `check_phase3_2.py`,
  `check_phase3_2_1.py`, `check_phase3_3.py` và `scripts/check_dataset_manifest.py`.
- Kaggle CUDA harness cho Phase 1 và Phase 3 (`kaggle-phase1/`, `kaggle-phase3/`,
  `scripts/build_kaggle_notebook.py`) cùng evidence tại `evidence/kaggle-phase3-pass/`.

### Limitations

- Dummy model không mang bất kỳ tuyên bố nghiên cứu nào.
- Model thật vẫn thuộc Phase 4.
- Phase 3 chưa đóng: P3.1 release closure và P3.4 contact-rich dynamic grasp còn mở.
- `QDGrasp-Scene-Tiny` được review trong scope micro fixtures; external real-source
  roots chưa cấu hình nên không có source-scale compatibility claim.
- CLI `qdgrasp data generate/audit` (P3-15) chưa được triển khai; dataset được sinh
  bằng `scripts/generate_dgn_open_tiny.py` và `scripts/generate_scene_tiny.py`.

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
