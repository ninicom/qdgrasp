---
document_id: ADR-0002
document_type: decision
title: Chuyển DexGrasp sang Apache-2.0 bằng triển khai clean-room
status: accepted
date: 2026-08-22
revises: none
supersedes: REV-20260822-005
---

# ADR-0002 — Apache-2.0 và biên clean-room

## Bối cảnh

Maintainer muốn DexGrasp có thể dùng cho cá nhân, nghiên cứu, cộng đồng và thương
mại với mức ràng buộc thấp, đồng thời giữ trải nghiệm đơn giản và hệ sinh thái mở
như các framework ML phổ biến.

Audit tại `HEAD 1f2d138` cho thấy đây chưa phải một implementation độc lập:

- 93/93 file tracked trong `dexgrasp/` có đường dẫn tương ứng ở snapshot
  Ultralytics đã pin; 87 file trùng byte, chiếm 94,01% dung lượng.
- 44.475/44.500 dòng text hiện tại khớp upstream; 90/91 file text trong payload
  mang header Ultralytics AGPL-3.0 (file còn lại là marker `py.typed` rỗng).
- Git ghi rõ commit `2bc248c` import nguyên trạng và `db0bfca` chỉ đổi namespace.
- DGN2 chưa được port vào product tree, nhưng upstream tuyên bố cả work và dataset
  dưới CC-BY-NC-4.0.

Vì vậy đổi root `LICENSE` không thể biến cây hiện tại thành permissive. “Apache
License 3.0” cũng không tồn tại: Apache Software Foundation xác định 2.0 là bản
hiện hành. `AGPL-3.0` là strong copyleft và là license đang dùng, không phải biến
thể Apache.

## Quyết định

1. Chọn `Apache-2.0` cho code do dự án sở hữu. Giữ đúng LICENSE/NOTICE,
   attribution, patent và trademark obligations; không quảng bá là “không có bản
   quyền”.
2. Freeze cây dẫn xuất hiện tại như hồ sơ AGPL, giữ nguyên notice/license. Sản
   phẩm permissive bắt đầu trong repository và Git history mới.
3. Không port implementation/config/data/checkpoint/assets từ Ultralytics hoặc
   DGN2 vào repository Apache. DGN2 chỉ được dùng như paper và oracle opt-in bên
   ngoài distribution; không dùng để train/fine-tune/distill official weights.
4. Dùng stack permissive đã audit ở exact pin:
   - PyTorch + Lightning Fabric cho runtime/training;
   - code Apache-2.0 chọn lọc từ GraspGenX cho grasp/conditioning khi phù hợp;
   - Pointcept MIT hoặc implementation từ paper cho point encoder;
   - rectified-flow MIT hoặc implementation trực tiếp từ công thức;
   - MuJoCo/MJX Apache-2.0 cho data generation/evaluation;
   - MuJoCo Menagerie theo license từng model cho hand assets.
5. Viết mới joint/qpos head, canonical schemas, façade/CLI, checkpoint bundle,
   robot contract và simulator protocol. GraspGenX chỉ sinh wrist pose 6/9 chiều
   nên không phải drop-in replacement cho DGN2 dexterous articulation.
6. Tự tạo `DGN-Open` từ procedural/CC0 objects và permissive hand assets. GSO/YCB
   CC-BY-4.0 chỉ nằm ở pack/weight track tách riêng có attribution.
7. Mọi public code, data, weights và assets có manifest/provenance độc lập. CI
   chặn copyleft, non-commercial, no-derivatives, custom/no-license và unknown
   khỏi core release.

## Nguồn chính thức đã kiểm

| Nguồn | Phạm vi | License/evidence | Cách dùng |
| --- | --- | --- | --- |
| Apache Software Foundation | License đích | `https://apache.org/licenses/` — Apache-2.0 là bản hiện hành | License code dự án mới |
| Ultralytics | Cây hiện tại | `https://www.ultralytics.com/license` — AGPL-3.0/Enterprise | Chỉ archive/reference; không port |
| DexGraspNet2 | Work và dataset | `https://github.com/PKU-EPIC/DexGraspNet2` — CC-BY-NC-4.0 | Paper/oracle cục bộ |
| GraspGenX `b942909` | Code | `https://github.com/NVlabs/GraspGenX/blob/b9429097728cb1c430dd78b92edf17ba318aad03/LICENSE` — Apache-2.0 | Chỉ file code đã audit; weights/data/assets riêng |
| Lightning | Training/runtime | `https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE` — Apache-2.0 | Engine substrate |
| Pointcept | Point-cloud code | `https://github.com/Pointcept/Pointcept/blob/main/LICENSE` — MIT | Pin/file audit trước reuse |
| rectified-flow | Flow code | `https://github.com/lqiang67/rectified-flow` — MIT | Ưu tiên công thức/implementation nhỏ |
| MuJoCo | Physics | `https://github.com/google-deepmind/mujoco/blob/main/LICENSE` — Apache-2.0 | Generator/evaluator |
| MuJoCo Menagerie | Robot models | `https://github.com/google-deepmind/mujoco_menagerie` — per-model license | Chỉ model permissive, giữ license thư mục |
| GSO/YCB | Object models | CC-BY-4.0 theo trang dataset chính thức | Optional attributed pack, không phải CC0 core |

License của repository không tự động bao phủ pretrained weights, dataset,
submodule, runtime download hoặc từng asset. Exact commit, transitive dependency,
file header và model/data card vẫn phải qua M0 gate.

## Clean-room boundary

- Nhóm đặc tả chỉ ghi behavior, tensor/schema contract, toán học từ paper và
  acceptance tests. Không đưa code/config text AGPL/NC vào đặc tả.
- Nhóm implementation chỉ dùng đặc tả và source allowlist. Mỗi file mới khai báo
  author/source/SPDX; mọi port permissive ghi original path/commit/hash.
- Nếu một người kiêm cả hai vai trò, project tăng kiểm soát bằng provenance log,
  review độc lập, exact-hash scan, similarity review và yêu cầu giải trình mọi
  match đáng kể. Similarity scan không tự chứng minh quyền sử dụng.
- Repository Apache không chứa `.references/` bị hạn chế; oracle chạy qua path do
  người dùng cung cấp và nằm ngoài wheel/sdist/container/test fixtures.

## Hệ quả

- Mất toàn bộ công sức trim Ultralytics như product code; các commit vẫn có giá
  trị làm bằng chứng lịch sử và đặc tả nhu cầu.
- Thời gian M1 tăng vì phải viết framework spine mới, nhưng M2/M3 chưa bắt đầu nên
  đây là thời điểm đổi hướng có chi phí thấp nhất.
- Không có dataset permissive drop-in 1:1 cho DGN2; data generation trở thành
  hạng mục bắt buộc và quyết định chất lượng official weights.
- Apache-2.0 cho phép downstream đóng source và thương mại; dự án chọn adoption
  rộng thay vì ép mọi cải tiến quay lại như AGPL.
- Quyết định này là đánh giá kỹ thuật/provenance, không phải ý kiến pháp lý. Full
  independent review vẫn là release gate trước public beta.
