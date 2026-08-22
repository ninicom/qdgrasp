---
document_id: ARCH-REF-20260822-002
document_type: reference_record
title: Bộ kiến trúc tham chiếu đa dạng cho QDGrasp
status: accepted
date: 2026-08-22
necessity: N2
impact: Giảm rủi ro đồng thuận sai bằng cách đối chiếu ba pipeline độc lập với các biểu diễn và mục tiêu khác nhau.
revision_reason: Maintainer yêu cầu thêm nhiều kiến trúc public để tránh phụ thuộc một tác giả hoặc một cách triển khai.
source_lock: architecture_references.lock.yaml
---

# Bộ kiến trúc tham chiếu đa dạng

Đã clone thêm ba repository ở commit cố định dưới
`.references/architectures/`. Chúng được dùng để đối chiếu thiết kế, không phải
nguồn runtime hay nguồn checkpoint của QDGrasp.

| Repository | License code | Ý tưởng độc lập | Giá trị tham khảo |
| --- | --- | --- | --- |
| [DexDiffuser](https://github.com/YuLiHN/DexDiffuser) | MIT | DexSampler + DexEvaluator + refinement trên partial point cloud | Tách generator/evaluator, denoising và refinement |
| [CEDex-Grasp](https://github.com/GeorgeWuzy/CEDex-Grasp) | MIT | human-like contact map → robot remapping → optimization | Cross-embodiment, contact-first, adaptation sang hand mới |
| [GraspGenX](https://github.com/NVlabs/GraspGenX) | Apache-2.0 | swept-volume conditioning, cross-embodiment 6-DOF diffusion | Profile registry, conditioning và zero-shot gripper design |

## Đối chiếu để tránh sao chép lỗi

- DexDiffuser phù hợp làm đối chứng cho pipeline diffusion nhưng phụ thuộc các
  asset/data/checkpoint bên ngoài; không xem các tệp đó là MIT.
- CEDex hữu ích cho contact representation nhưng checkout có `.pt`, data và
  assets của nhiều nguồn; toàn bộ các artifact này bị chặn khỏi release.
- GraspGenX là baseline permissive tốt cho cross-embodiment gripper, nhưng là
  bài toán 6-DOF gripper, không thay thế joint-state dexterous hand. Checkpoint
  dùng NVIDIA Open Model License riêng.
- Các điểm giao nhau chỉ được dùng làm hypothesis. QDGrasp phải chọn bằng
  ablation/contract tests giữa direct flow, contact-first và evaluator-guided
  refinement; không hợp thức hóa thiết kế chỉ vì xuất hiện ở một repository.

Manifest, commit, SPDX và ranh giới artifact nằm trong
`architecture_references.lock.yaml`.
