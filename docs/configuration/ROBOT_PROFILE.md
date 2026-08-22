---
document_id: QDGRASP-ROBOT-V2
document_type: registry
title: Registry schema cấu hình robot profile QDGrasp v2
version: 1.0.0
status: active
date: 2026-08-22
revises: none
related_plan: PLAN-V2
---

# Registry schema cấu hình robot profile QDGrasp v2

Tài liệu này mô tả contract cấu hình robot profile `qdgrasp/robot/v2` do Phase 2
tạo ra trong `qdgrasp/robot/schema.py`. Schema này sống cạnh `qdgrasp/robot/v1`
(được giữ nguyên cho dummy profile) và mở rộng cho robot thật với URDF/MJCF, mesh
resolver, kinematic tree, semantic links, coupling/mimic policy và MuJoCo
actuators.

## 1. Nguyên tắc

- Mỗi document YAML bắt buộc có trường `schema: qdgrasp/robot/v2`.
- Unknown key và dead key là lỗi (`extra="forbid"`).
- Parser dùng `yaml.safe_load` cộng Pydantic validation nghiêm ngặt.
- Document đã validate là immutable và có `content_hash()` SHA-256 ổn định.
- Semantic links (`palm_link`, `fingertip_links`, `contact_links`) bắt buộc phải
  khai báo tường minh; parser **từ chối suy đoán im lặng**.
- Mọi profile có cờ `release_blocked: true` (như Barrett hand hoặc fixture từ
  `dexsuite_dex_urdf`) bị chặn khỏi mọi public release.

## 2. Document `qdgrasp/robot/v2`

| Key | Type | Default | Ràng buộc |
| --- | --- | --- | --- |
| `schema` | str | bắt buộc | phải là `qdgrasp/robot/v2` |
| `name` | str | bắt buộc | tên profile hiển thị |
| `format` | str | `"mjcf"` | `"mjcf"` hoặc `"urdf"` |
| `source_asset` | str | bắt buộc | đường dẫn tương đối tới file XML/URDF |
| `palm_link` | str | bắt buộc | link/body gốc của palm pose |
| `base_link` | str \| null | `null` | link gốc của toàn bộ cánh tay/bàn tay |
| `wrist_link` | str \| null | `null` | link khớp cổ tay nếu có |
| `fingertip_links` | list[str] | `[]` | danh sách có thứ tự link đầu ngón tay |
| `contact_links` | list[str] | `[]` | danh sách các link tham gia tiếp xúc |
| `joints` | list[str] | bắt buộc | danh sách có thứ tự các actuated named joints |
| `joint_limits` | map[str, [float, float]] | bắt buộc | phủ đúng tập `joints`, hữu hạn, `lower < upper` |
| `mimic_joints` | map[str, MimicSpec] | `{}` | khai báo quan hệ mimic / coupling |
| `actuators` | map[str, ActuatorSpec] | `{}` | tham số điều khiển và squeeze |
| `mesh_root` | str \| null | `null` | thư mục gốc tìm mesh bổ sung |
| `package_roots` | map[str, str] | `{}` | ánh xạ package ROS `package://<pkg>/` |
| `frame` | str | `"palm"` | frame quy chiếu |
| `release_blocked` | bool | `false` | cờ chặn phát hành công khai |
| `provenance` | map[str, Any] | `{}` | metadata nguồn, commit, license SPDX |

## 3. Mimic và Actuator Specs

### MimicSpec

| Key | Type | Default | Ý nghĩa |
| --- | --- | --- | --- |
| `target_joint` | str | bắt buộc | joint chủ động điều khiển joint này |
| `multiplier` | float | `1.0` | hệ số tỷ lệ $q_{\text{mimic}} = m \cdot q_{\text{target}} + c$ |
| `offset` | float | `0.0` | độ lệch $c$ |

### ActuatorSpec

| Key | Type | Default | Ý nghĩa |
| --- | --- | --- | --- |
| `joint` | str \| null | `null` | tên joint được gán actuator |
| `type` | str | `"position"` | loại điều khiển (`position`, `velocity`, `motor`) |
| `kp` | float | `1.0` | hệ số proportional gain |
| `kv` | float | `0.1` | hệ số derivative/damping gain |
| `ctrl_range` | [float, float] \| null | `null` | khoảng giá trị control tín hiệu |
| `force_range` | [float, float] \| null | `null` | khoảng lực/mô-men giới hạn |
| `squeeze_target` | float \| null | `null` | giá trị mục tiêu khi thực hiện squeeze |

## 4. Presets có sẵn

Phase 2 phát hành 3 presets chuẩn:
1. `leap_hand.yaml`: LEAP Hand v2 (16 actuated joints, Menagerie MJCF).
2. `wonik_allegro.yaml`: Wonik Allegro Hand v2 (16 actuated joints, Menagerie MJCF / normalized URDF).
3. `shadow_hand.yaml`: Shadow Hand E3M5 v2 (20 actuated joints từ $nq=24$, 4 mimic joints, Menagerie MJCF).

Các preset dùng `source_asset: asset://…`, không trỏ tới đường dẫn workspace.
Wheel không mang MJCF/mesh robot; trước khi gọi `RobotSpec.from_config`, đặt
`QDGRASP_ROBOT_ASSETS_ROOT` tới checkout `robot-assets` khớp manifest. Thiếu
biến này là lỗi cấu hình có hướng dẫn, không phải fallback sang asset khác.

`qdgrasp/assets/derived/*.urdf` là artifact normalization/provenance, không phải
robot bundle tự chứa: mesh của nó vẫn thuộc external asset checkout. Không dùng
file đó trực tiếp từ bare wheel; một asset bundle phân phối riêng sẽ cần review
license/provenance trước khi được thêm.
