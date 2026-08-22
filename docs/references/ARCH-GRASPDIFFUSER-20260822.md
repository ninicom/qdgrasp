---
document_id: ARCH-REF-20260822-001
document_type: reference_record
title: GraspDiffuser — kiến trúc tham chiếu MIT cho dexterous grasp
status: accepted
date: 2026-08-22
necessity: N2
impact: Cung cấp code/structure tham chiếu cô lập để triển khai QDGrasp mới mà không phụ thuộc DGN2.
revision_reason: Maintainer yêu cầu một kiến trúc dexterous-grasp public có license mở để giảm rủi ro triển khai từ đầu.
source_lock: architecture_references.lock.yaml
---

# GraspDiffuser — kiến trúc tham chiếu

## Kết luận

Đã clone GraspDiffuser tại commit `ae085627ab0108527255a3da8f6728010a5f1516`
vào `.references/architectures/GraspDiffuser/`. Upstream công bố code dưới MIT;
repository cũng ghi rõ các phần kế thừa từ QDGrasp Anything là MIT và các thành
phần như GraspQP, dataset, robot assets và dependencies có license riêng.

Nguồn này phù hợp để đọc cấu trúc module, diffusion scheduler, scene encoder,
data collation, normalization, Lightning training loop và Hydra YAML presets.
Nó không phải dependency runtime của `qdgrasp/`.

## Ánh xạ tham khảo sang QDGrasp

| GraspDiffuser | QDGrasp mới |
| --- | --- |
| scene point-cloud encoder | object encoder + optional contact/normal features |
| conditional DDPM | flow-matching/diffusion backend theo schema riêng |
| hand pose representation | palm SE(3) + named joints + HandGraph keypoints |
| selectable hand model | `RobotSpec`/URDF-MJCF adapter |
| Hydra config groups | YAML preset/stage schema của QDGrasp, không sao chép parser |
| Lightning module | engine/training façade viết mới |

## Ranh giới sử dụng

- Chỉ đọc hoặc dùng làm mẫu để viết implementation độc lập.
- Không dùng upstream checkpoint, dataset, GraspQP, robot asset hoặc dependency
  ngoài như thể chúng thuộc MIT.
- Nếu sao chép một đoạn code cụ thể, phải ghi attribution/NOTICE và mở revision
  provenance riêng; mặc định ưu tiên viết lại từ interface/toán học độc lập.
- Không đưa checkout này vào package, import path, test release hay official
  checkpoint của QDGrasp.

Manifest và commit pin nằm trong `architecture_references.lock.yaml`.
