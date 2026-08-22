---
document_id: REV-20260823-001
document_type: revision
title: Khóa Literature Snapshot và Đánh dấu Data Correctness P3
version: 1.0.0
status: active
date: 2026-08-23
revises: SESSION-20260822-022
related_plan: ROADMAP-P3.1-001
---

# Khóa Literature Snapshot và Đánh dấu Data Correctness P3 (Module P3.1-00)

Revision này chính thức khởi động Phase 3.1 nhằm giải quyết các lỗ hổng logic nghiêm trọng được phát hiện tại phiên làm việc `SESSION-20260822-022`.

## 1. Khóa Literature Snapshot
Literature snapshot được khóa lại ở mức `2026-08-23`. Các lý thuyết/phương pháp sẽ được module hóa và triển khai bao gồm:
- **EquiDexFlow (06/2026)**: Tuple contact + lực, Coulomb-cone projection, lift mượt (smooth lift).
- **Transferring Contact, Not Just Motion (06/2026)**: Mô tả lực (N) và mô-men (N·m) theo từng ngón độc lập.
- **SynManDex (06/2026)**: Cơ chế kiểm định (admission) theo nhiều giai đoạn độc lập.
- **SECOND-Grasp (05/2026)**: Xác định vùng tiếp xúc (surface region) và point-to-region IK.
- **BiDexGrasp (04/2026)**: Căn chỉnh (ranking) vùng theo đánh giá wrench (khả năng chịu lực) sơ bộ.

## 2. Invalidate Dataset Hiện Tại
Bản sinh dữ liệu hiện tại của `DGN-Open-Tiny` đã vi phạm các nguyên tắc vật lý (Kinematic Teleportation) và sai lệch về Analytical IK. Qua đó, 144 mẫu trong bản phân phối hiện tại bị đánh dấu là **INVALIDATED** (chứa 0 positive samples dù module báo false positive closure).

**Hash của dataset bị invalidated:**
- `datasets/dgn-open-tiny/dataset_manifest.json`: `dcff16f83f5c397cc2b478efda8f8ac837fcad836c234017d234f4d5548428a1`
- Toàn bộ 6 shard liên quan.

Các file sẽ được regenerate (tạo lại hoàn toàn) từ các thư mục staging an toàn sau khi toàn bộ Phase 3.1 pass các Regression Testing Gate.

## 3. Điều chỉnh Luận điểm P3
Những công bố về thành tựu của P3 tại `SESSION-20260822-022` tạm thời bị gỡ bỏ giá trị nghiệm thu do không đảm bảo tính toàn vẹn (Correctness). Data Layer sẽ tiếp tục làm nền móng nhưng được siết chặt qua các "Typed Contracts".
