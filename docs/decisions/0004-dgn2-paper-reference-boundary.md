---
document_id: ADR-0004
document_type: decision
title: Biên DGN2 chỉ tham chiếu paper, không phải oracle hay dependency
status: accepted
date: 2026-08-22
revises: ADR-0002
supersedes: none
necessity: N2
impact: Clarify the DGN2 paper/reference boundary for model, data, checkpoint and release documentation without changing the accepted clean-room decision.
revision_reason: Maintainer xác nhận DGN2 chỉ là bài báo tham chiếu nguyên lý grasping; toàn bộ code, schema, data, simulator và checkpoint của DexGrasp phải phát triển mới.
---

# ADR-0004 — DGN2 paper/reference-only

## Quyết định

DGN2 chỉ được dùng như tài liệu học thuật để hiểu bài toán và nguyên lý
dexterous grasping. Nó không phải oracle, runtime dependency, nguồn nhãn hay
baseline artifact bắt buộc của DexGrasp.

DexGrasp phát triển mới toàn bộ:

- model, module, YAML schema và parser;
- `RobotSpec`, HandGraph, FK/physics protocol và simulator adapter;
- data generator, dataset schema, labels và splits;
- checkpoint, official weights và benchmark artifacts.

Không sử dụng hoặc đóng gói code, config, dataset, labels, URDF/STL/mesh,
checkpoint, pretrained initialization, distillation target hoặc implementation
expression từ DGN2. Không copy figure/prose/pseudocode; chỉ trích dẫn paper và
viết lại thuật toán từ yêu cầu/toán học độc lập.

## Ranh giới license

AGPL-3.0 (hoặc license đích của code phát hành sau khi maintainer chốt) áp dụng
cho code do DexGrasp sở hữu. Điều đó không thay đổi license CC-BY-NC của DGN2
work/dataset và không cấp quyền redistribute DGN2 artifacts.

Robot/object assets và third-party dependencies vẫn được kiểm tra theo manifest
riêng. Asset permissive có thể dùng nếu exact source, SPDX, attribution và hash
đã qua gate; asset DGN2 không được đưa vào core/release.

## Cách dùng hợp lệ

Có thể tham khảo các khái niệm như object point cloud, graspness, palm pose,
joint state, collision/squeeze/lift evaluation và robot-aware kinematics. Mọi
schema, module boundary, implementation, data và kết quả của DexGrasp phải được
thiết kế/triển khai/đo độc lập.

## Hệ quả

- Published results của DGN2 chỉ là literature context; nếu so sánh, phải ghi
  protocol khác biệt và `comparability: limited`.
- Không cần tải DGN2 checkpoint/dataset để build hoặc test release.
- DGN-Open và permissive robot/object corpus là nguồn duy nhất cho official
  weights.
- Decision này làm rõ wording của ADR-0002 mà không rewrite record accepted cũ.
