---
document_id: ADR-0008
document_type: decision
title: Tạm dừng Shadow Hand khỏi phạm vi hoạt động mặc định
status: accepted
date: 2026-08-27
revises: none
supersedes_in_part:
  - ADR-0003
  - PLAN-V2@4.0.0
latest_revision_record: docs/revisions/REV-20260827-009-temporary-shadow-hand-pause.md
owner: project-maintainer
---

# ADR-0008 — Tạm dừng Shadow Hand

## Bối cảnh

Shadow Hand E3M5 đã được đưa qua parser, `RobotSpec`, fixed-tendon transmission,
P2/P3.2/P3.3 fixtures và nhiều chẩn đoán P3.4. Tuy nhiên cấu hình dynamic grasp
cho Shadow hiện có chi phí cao và chưa ổn định:

- 24 joint state nhưng 20 actuator, nhiều fixed tendon cần semantics riêng;
- recipe inactive-finger cũ tạo legitimate self-contact trong palm;
- miền posture an toàn và miền tracking đã thử chưa cho positive hợp lệ;
- validator/controller cần thêm projected-target hoặc safe-set semantics;
- việc tiếp tục bắt mọi backend/dataset/RL gate chờ Shadow đang chặn tiến độ
  LEAP và Allegro.

Đây là vấn đề chi phí và độ khó cấu hình ở trạng thái hiện tại, không phải kết
luận rằng Shadow hardware, Menagerie asset hoặc mô hình underactuated về nguyên
tắc không thể hỗ trợ.

## Quyết định

1. Tập hand hoạt động mặc định từ ngày 2026-08-27 là:
   `leap_hand` và `wonik_allegro`.
2. `shadow_hand` chuyển sang trạng thái **paused/experimental**. Không dùng nó
   trong generation, training, checkpoint, release dataset, simulator/backend
   selection gate hoặc RL readiness gate mới.
3. Không xóa `shadow_hand.yaml`, parser, transmission code, tests chuyên biệt,
   asset provenance hay evidence cũ. Chúng được giữ cho compatibility, audit và
   khả năng khôi phục.
4. Normal/default workload không được tự chọn Shadow. Chỉ được chạy Shadow trong
   một diagnostic/resumption task được maintainer cho phép rõ ràng; kết quả đó
   không tự trở thành release evidence.
5. Mọi tài liệu/report/evidence đã đóng vẫn bất biến và tiếp tục mô tả đúng phạm
   vi ba hand tại revision lịch sử của nó. Không rewrite lịch sử thành hai hand.
6. P3.4 contract ba-hand hiện tại **không đạt và tạm dừng**. Không được bỏ Shadow
   khỏi gate cũ rồi ghi P3.4 `complete`.
7. P3.4.2 closure contract được giữ như thiết kế khôi phục Shadow nhưng execution
   chuyển `superseded/paused` trong thời gian ADR này có hiệu lực.
8. P3.5 và công việc mới sau quyết định này dùng compatibility matrix **2/2
   active hand**. Claim phải ghi rõ `active_hands=[LEAP, Allegro]` và
   `paused_hands=[Shadow]`; không được gọi kết quả đó là three-hand coverage.

## Phạm vi áp dụng

Quyết định này áp dụng cho:

- living `PLAN.md`, project roadmap và các phase đang active;
- default dataset generation/training/evaluation configuration;
- release manifests/checkpoints tạo sau ngày quyết định;
- backend selection, cloud GPU evidence và RL environment mới.

Quyết định này không làm mất hiệu lực bằng chứng đã pass cho P2/P3.2/P3.3 trên
exact historical commit. Nó chỉ ngăn kết quả cũ bị suy thành cam kết tiếp tục
phủ Shadow trong release mới.

## Điều kiện mở lại Shadow

Shadow chỉ trở lại active scope bằng một ADR/revision mới sau khi có tối thiểu:

1. maintainer phê duyệt mục tiêu và resource budget;
2. exact asset/profile/controller/protocol lock;
3. diagnostic tách requested/projected/actuator tracking;
4. no-closure và swept-path self-contact safety pass;
5. ít nhất một dynamic positive qua support-release/lift/perturbation + CPU replay;
6. GPU/backend parity và regression với LEAP/Allegro;
7. independent review cho blast radius của việc kích hoạt lại.

Cho tới lúc đó, missing Shadow result phải mang trạng thái `not_applicable_paused`
hoặc `paused_by_ADR-0008`, không dùng `pass`, `zero`, `unsupported` hoặc
`not_run` mơ hồ.

## Hệ quả

- Tiến độ LEAP/Allegro và P3.5 không còn chờ Shadow configuration.
- P3.4 ba-hand không thể đóng trong thời gian pause; nếu cần một dynamic-data
  release hai hand phải tạo scope/plan successor riêng, không tái sử dụng verdict
  P3.4 cũ.
- Cross-embodiment architecture vẫn variable-topology và không hard-code hai
  hand; pause chỉ thay active corpus/gate.
- Resume Shadow là thay đổi N2/N3 cần revision, rerun affected gates và review.
