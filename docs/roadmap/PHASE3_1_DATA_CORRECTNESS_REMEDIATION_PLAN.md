---
document_id: ROADMAP-P3.1-001
document_type: plan
title: Kế hoạch Phase 3.1 — Data Correctness Remediation
version: 1.3.0
status: active
date: 2026-08-23
revises: ROADMAP-P3.1-001@1.2.0
related_plan: ROADMAP-P3-001
literature_cutoff: 2026-08-23
---

# Kế hoạch Phase 3.1 — Data Correctness Remediation

P3.1 là remediation nằm giữa P3 và P4. Nó không thêm model, training objective
hay benchmark mới; mục tiêu duy nhất là chứng minh dữ liệu đầu vào của P4 có
contact hình học hợp lệ, IK có ý nghĩa, nhãn vật lý đo đúng đại lượng đã khai báo
và artifact phát hành thực sự được Git/provenance gate bảo vệ.

P4 bị chặn cho tới khi toàn bộ gate ở §9 pass. `DGN-Open-Tiny` hiện tại không
được dùng làm bằng chứng overfit hoặc chất lượng model.

## 1. Baseline và kết luận audit

- Baseline code: `40d37b6` trên `feature/phase3-data-layer`.
- Session bị ảnh hưởng: `SESSION-20260822-022`.
- Plan gốc: `docs/roadmap/PHASE3_EXECUTION_PLAN.md`.
- File nháp không tracked `test_mocap.py` không thuộc P3.1 và phải được giữ
  nguyên trừ khi maintainer đưa nó vào scope bằng revision riêng.

| ID | Phát hiện tại baseline | Kết luận |
| --- | --- | --- |
| F-01 | `sample.py` đã lấy approach point trên mesh, nhưng tạo từng `target_contact` bằng `surface_point + normal(0, 0.015)` trong 3D | Contact target vẫn có thể nằm trong vật hoặc ngoài không gian; defect chỉ được sửa một phần |
| F-02 | Chẩn đoán box 4 cm, 8 candidate/hand: 0/104 target nằm trên surface trong tolerance `1e-6 m`; khoảng cách lớn nhất `0.0371 m` | Blocker correctness |
| F-03 | `solve_dls_ik` dùng autograd thay finite difference nhưng API vẫn ép batch 1, generator gọi tuần tự, objective chỉ có vị trí và candidate không mang target normal | Blocker chất lượng; batching là blocker throughput trước khi scale |
| F-04 | Hai candidate đầu của cả LEAP/Allegro/Shadow trong chẩn đoán đều không hội tụ, error tổng `0.108–0.184 m`; generator vẫn đưa nghiệm không hội tụ qua filter/simulation | Blocker correctness |
| F-05 | `labeling.py` đã chuyển lift/perturbation sang mocap body + weld constraint | Hướng sửa đúng theo MuJoCo; không quay lại sửa `model.body_pos` giữa simulation step |
| F-06 | Code gọi tiêu chí là “force closure” nhưng chỉ đếm contact/link; không đọc `mj_contactForce`, không lập grasp wrench và chỉ đo penetration ở frame cuối | Nhãn success chưa chứng minh lực tiếp xúc hay force closure |
| F-07 | Sáu shard hiện tại đều có `positive_samples=0` | Dataset không đủ cho P4 overfit/classification gate |
| F-08 | `datasets/dgn-open-tiny/` bị `/datasets/` và `*.pt` ignore; manifest không được `git ls-files` nhận | Artifact chưa phải release tái lập trong repository |
| F-09 | Unit test hiện chỉ kiểm shape, joint limit và hai lần chạy giống nhau; các test đó vẫn pass khi toàn bộ nhãn là âm | Gate hiện tại không phát hiện defect logic |

## 2. Literature snapshot 2026 — bản đọc phục vụ triển khai

Literature được rà tới `2026-08-23`. Không tìm thấy paper tháng 7–8/2026 khớp
trực tiếp hơn với bài toán **single-hand procedural grasp dataset + contact/force
certification**; nhóm nguồn mới nhất và sát nhất là tháng 6/2026. Phần lớn là
preprint, vì vậy P3.1 chỉ kế thừa representation, decomposition và invariant có
thể tự kiểm chứng. Success rate của paper không được dùng làm threshold của dự
án vì object set, robot, simulator và protocol khác nhau.

### 2.1 Từ điển tối thiểu để đọc các paper

- **Contact point** `p`: vị trí ngón chạm bề mặt. Một điểm gần vật không phải là
  contact nếu nó không nằm trên surface.
- **Inward normal** `n`: hướng từ mặt ngoài đi vào vật tại `p`. Normal gắn với
  đúng face/contact; project point sang face khác mà giữ normal cũ tạo dữ liệu
  tự mâu thuẫn.
- **Contact force** `f`: được tách thành thành phần pháp tuyến `f_n` và tiếp tuyến
  `f_t`. Với ma sát Coulomb, lực hợp lệ phải thỏa `f_n >= 0` và
  `||f_t|| <= mu * f_n`. Nói đơn giản: ngón có thể đẩy vào vật và ma sát chỉ giữ
  được một lượng lực trượt hữu hạn; ngón không thể “kéo” vật qua một contact đơn
  phương.
- **Wrench**: cặp lực và mô-men 6D tác động lên vật,
  `w = [f, (p - c) x f]`, với `c` là tâm quy chiếu. Hai grasp có cùng tổng lực
  nhưng mô-men rất khác nhau nếu vị trí contact khác nhau.
- **Force closure**: tập contact có thể tạo wrench chống lại nhiễu theo mọi hướng
  trong giả định ma sát đã chọn. Đây là chứng nhận tĩnh; nó chưa chứng minh bàn
  tay có thể đi tới grasp, nâng vật, hay không làm vật trượt trong chuyển động.
- **Ba tầng kết quả không được nhập làm một**:
  `proposal` chỉ là ứng viên hình học; `static feasible` đã qua IK/collision/
  force-closure; `dynamic valid` đã sống sót qua squeeze/lift/disturbance trong
  simulator. Chỉ tầng cuối mới được xét làm nhãn success của dataset.

### 2.2 EquiDexFlow — biểu diễn contact và force như một cấu trúc thống nhất

Nguồn: [EquiDexFlow, 10/06/2026](https://arxiv.org/abs/2606.12728).

**Bài toán.** Nhiều phương pháp chỉ sinh wrist pose hoặc joint angle rồi hy vọng
physics filter sửa phần còn lại. EquiDexFlow xem grasp là một cấu trúc gồm wrist
pose, joint configuration, contact theo từng ngón, normal/contact frame và lực
contact. Điểm quan trọng với P3.1 không phải flow network mà là: các trường này
phải đi cùng nhau và phải biến đổi nhất quán khi object pose thay đổi.

**Pipeline chính.** Decoder dự đoán offset contact theo từng ngón, snap mềm điểm
đó về surface, dự đoán normal và lực tương ứng. Lực được biểu diễn trong local
contact frame rồi project vào Coulomb cone; tổng wrench được kiểm tra khả năng
cân bằng gravity. Khi chuyển sang LEAP, paper dùng IK theo từng ngón và refinement
để nghiệm không nằm sát joint limit. Protocol lift của paper dùng một prismatic
joint được actuate theo smoothstep, tức vận tốc bằng không ở hai đầu, thay vì
teleport root body.

**Điều dễ hiểu sai.** “Project contact lên surface” chưa đủ. Nếu normal dùng để
tạo force khác normal của face sau projection, lực tưởng nằm trong friction cone
có thể thực tế nằm ngoài cone. Chính paper cũng nêu chất lượng normal và local
refinement là giới hạn; thử nghiệm hardware còn ít và open-loop. Vì vậy P3.1
không nhận output của một module là đúng chỉ vì shape hợp lệ.

**Quyết định cho P3.1.** Đưa tuple
`(point, normal, force_hint, finger_id)` vào contract chung và tạo module
`contact_force` độc lập để project/check force cone. Kế thừa smooth lift và
joint-margin refinement. Không triển khai flow model, equivariant network hoặc
claim chất lượng tương đương paper.

### 2.3 Transferring Contact, Not Just Motion — đại lượng phải có đơn vị vật lý

Nguồn: [Transferring Contact, Not Just Motion, 14/06/2026](https://arxiv.org/abs/2606.15516).

**Bài toán.** Cùng một giá trị actuator effort không có cùng nghĩa trên hai bàn
tay khác morphology, transmission hoặc motor. Paper hiệu chuẩn command thành
torque khớp theo `N·m`, dùng Jacobian để suy ra lực đầu ngón theo `N`, rồi điều
khiển bằng giao diện force-position chung. Trạng thái không chỉ có pose mà còn có
fingertip position, force và descriptor mô-men không gian theo từng ngón.

**Điểm liên quan trực tiếp.** Retargeting của paper kết hợp vị trí đầu ngón,
hướng chỉ của ngón và hình dạng toàn ngón, có regularization theo thời gian. Điều
này giải thích vì sao IK position-only có thể chạm đúng tọa độ nhưng tạo contact
dễ trượt: hướng tiếp xúc và cách tải lực vẫn sai. Nó cũng cho thấy
`contact_count=4` không phải metric có thể so sánh LEAP, Allegro và Shadow.

**Giới hạn.** Mỗi hardware cần system identification riêng; lỗi calibration lan
thẳng sang force estimate. Kết quả dựa trên số rollout nhỏ, chưa xử lý articulated
object và gain còn phụ thuộc hand/task. P3.1 chạy simulation nên không được giả
vờ đã giải quyết bài toán sim-to-real calibration.

**Quyết định cho P3.1.** Physics output phải lưu force `N`, torque `N·m`,
per-finger identity và aggregate wrench theo cùng convention cho cả ba hand.
Normal-alignment được đưa vào IK. Không thêm MANO latent, learned policy, sensor
calibration hay hybrid hardware controller.

### 2.4 SynManDex — proposal không phải grasp, force closure chưa phải execution

Nguồn: [SynManDex, 08/06/2026](https://arxiv.org/abs/2606.09798).

**Bài toán.** Human-like pre-grasp có thể đẹp nhưng floating, xuyên vật hoặc
không reachable với robot thật. SynManDex tách pipeline thành: human proposal →
robot-native contact/force-closure optimization → admission → trajectory. Một
candidate chỉ được nhận khi lần lượt qua collision, force closure, arm-hand IK và
lift trên chính robot model.

**Điểm quan trọng nhất.** Paper báo riêng force-closure success, IK/lift
admission, policy success, transition success và hardware success. Các con số
không giống nhau: một grasp tĩnh tốt vẫn có thể trượt trong approach/transition.
Force-closure dùng grasp map và friction cone rời rạc; tác giả cũng nói rõ score
admission không phải bảo đảm vật lý tuyệt đối, nên ứng viên sống sót vẫn phải
chạy simulator dưới gravity.

**Quyết định cho P3.1.** Orchestrator phải tạo status/reason riêng cho
`proposal_rejected`, `ik_rejected`, `collision_rejected`,
`static_force_rejected`, `dynamic_rejected`, `accepted`. Manifest lưu count của
từng stage. Không được dùng một boolean `success` trung gian rồi mất nguyên nhân.
Không thêm human pre-grasp, VLM task generation, bimanual policy hay RL.

### 2.5 SECOND-Grasp — region trước, contact chính xác sau

Nguồn: [SECOND-Grasp, 13/05/2026](https://arxiv.org/abs/2605.13117).

**Bài toán.** Ép mỗi fingertip tới một tọa độ ngẫu nhiên cố định làm IK dễ bị quá
ràng buộc. SECOND-Grasp tạo semantic surface region từ nhiều view, back-project
về đúng object surface, mở rộng seed theo local convexity, rồi chia region cho
thumb và nhóm ngón còn lại để tạo opposition.

**Cách IK khác baseline.** Thay vì bắt fingertip `i` tới đúng một điểm `y_i`,
objective dùng khoảng cách từ fingertip tới vùng hợp lệ của ngón đó:
`min_{y in R_i} ||tip_i(q) - y||²`. Solver được quyền chọn điểm reachable tốt
nhất bên trong region; contact chính xác chỉ được chốt và lưu sau khi IK hội tụ.
Đây là cách giảm over-constraint mà vẫn giữ contact trên bề mặt.

**Giới hạn.** Semantic/VLM stack của paper không cần thiết cho procedural object
và paper chưa đánh giá đầy đủ arm-hand platform. Local-convex heuristic cùng các
constant của object set paper không được copy nguyên xi sang mesh có scale khác.

**Quyết định cho P3.1.** Tạo module `region_opposition` không semantic: sinh vùng
bề mặt tất định, scale-normalized, phân thumb/opposing fingers; kết hợp solver
`region_dls`. Sau IK phải materialize exact point, face id và normal, không lưu
region mơ hồ làm ground truth.

### 2.6 BiDexGrasp — chọn region bằng khả năng tạo wrench

Nguồn: [BiDexGrasp, 08/04/2026](https://arxiv.org/abs/2604.06589).

**Bài toán.** Contact phân tán ngẫu nhiên có thể nằm trên surface nhưng vẫn cùng
đẩy một phía và không chống được mô-men. BiDexGrasp dùng contact position, inward
normal và friction cone để dựng grasp matrix; mỗi lực local được map thành wrench
6D. Khả năng chống sáu hướng nhiễu cơ bản được đánh giá bằng bài toán tối ưu lực,
rồi region tốt mới được dùng để khởi tạo/refine grasp.

**Pipeline.** Paper làm region-based initialization trước force-closure
optimization, sau đó mới kiểm collision/reachability. Năng lượng cho hai hand
được tách để tránh một hand rất mạnh che sự vô dụng của hand còn lại. Với dự án
single-hand, bài học tương ứng là không để palm hoặc một finger có lực lớn che
việc các finger còn lại không tạo opposition.

**Giới hạn.** Hằng số sampling, số point lân cận và tiêu chí cặp region của paper
phục vụ bimanual object set riêng. Paper vẫn quan sát failure do penetration,
floating và lateral slip; tối ưu wrench tĩnh không thay dynamic rollout.

**Quyết định cho P3.1.** Tạo module `wrench_guided`: rank anchor/region bằng
grasp-wrench feasibility scale-normalized, rồi mới sinh contact set. Static
certifier dùng chung grasp map/QP để tránh sampler tự chấm bài của chính nó bằng
một implementation khác convention. P3.1 không mở scope sang bimanual.

### 2.7 Tổng hợp thành kiến trúc, không thành khẩu hiệu “SOTA”

| Paper | Ý tưởng được cô lập thành module | Phép kiểm độc lập bắt buộc |
| --- | --- | --- |
| EquiDexFlow | force-conditioned contact tuple, cone projection, smooth lift | surface/normal consistency, cone residual, external static certifier |
| Transferring Contact | physical per-finger load descriptor | unit/convention test và tổng per-finger wrench |
| SynManDex | staged admission orchestrator | reason count và không bỏ qua stage |
| SECOND-Grasp | surface region + point-to-region IK | exact post-IK contact nằm trong region/surface |
| BiDexGrasp | wrench-guided region ranking | QP/GWS certificate và dynamic disturbance |

Representation chuẩn tối thiểu của P3.1 vì thế là `(contact point hoặc target
region, inward normal, admissible/contact force, finger/link identity,
provenance)`. Mỗi technique nằm sau interface chung và phải được ablate cùng
seed/budget; không dùng cụm “match SOTA” làm acceptance criterion.

Hai nguồn kỹ thuật vẫn có thẩm quyền dù không phải paper 2026:

- [MuJoCo modeling documentation](https://mujoco.readthedocs.io/en/latest/modeling.html#mocap-bodies)
  là nguồn chuẩn cho mocap body + regular dynamic body + weld equality. Baseline
  đã đi đúng hướng; P3.1 harden trajectory, force measurement và test.
- Các paper 2021–2023 trong plan version 1.0 chỉ còn historical background, không
  còn là nguồn chính cho quyết định P3.1.

## 3. Ranh giới phạm vi

### Thuộc P3.1

- Deterministic surface/contact-set sampling cho nhiều fingertip.
- Ba proposal recipe module hóa và ablation cùng budget.
- Contact normal và normal-alignment objective.
- Fixed-contact/region batched DLS-IK, convergence mask và joint-limit projection
  cho ba hand.
- MuJoCo mocap-weld trajectory có giới hạn, contact-force/wrench metrics và
  penetration theo toàn trajectory.
- Regenerate `DGN-Open-Tiny` có cả positive/negative samples.
- Narrow Git allowlist, tracked release gate và revision record sửa claim P3.

### Không thuộc P3.1

- Model/flow/encoder/loss của P4.
- Goal-conditioned RL policy của UniDexGrasp.
- Tái hiện nguyên bản DexGraspNet, Contact-GraspNet hoặc GraspIt!.
- Dataset lớn, distributed generation, benchmark throughput của P6.
- Thêm dependency mới hoặc thay environment lock nếu không có revision riêng.
- Sửa byte của session P3 đã đóng; correction phải qua revision record mới.

## 4. Thiết kế bắt buộc

### 4.0 Kiến trúc module và recipe được hỗ trợ

Không viết lại `sample.py`, `ik.py` và `labeling.py` thành ba “siêu hàm”. P3.1
tách pipeline theo ranh giới toán học; module chỉ biết input/output contract của
nó và không được gọi ngược hoặc bỏ qua stage khác:

```text
ObjectSurface
    -> ProposalStrategy
    -> KinematicSolver
    -> StaticCertifier
    -> DynamicValidator
    -> PipelineOutcome + stage evidence
```

Các contract bất biến đặt tại `qdgrasp/dataset/pipeline/contracts.py`:

- `ContactProposal`: target point/region, face id, inward normal, finger id,
  optional force hint và provenance của strategy;
- `KinematicSolution`: `q`, exact achieved contacts/normals, per-finger residual,
  convergence mask và reason;
- `StaticCertificate`: force solution, cone residual, object wrench, quality
  margin và pass/fail;
- `DynamicValidation`: trajectory metrics, per-finger measured loads, failure
  stage và pass/fail;
- `PipelineOutcome`: ghép evidence của bốn stage mà không làm mất intermediate
  rejection.

Module layout dự kiến:

| Nhóm | Module | Trách nhiệm duy nhất |
| --- | --- | --- |
| Proposal | `proposals/surface_fixed.py` | Reference sampler: exact deterministic surface points |
| Proposal | `proposals/region_opposition.py` | SECOND-style thumb/opposing-finger surface regions |
| Proposal | `proposals/wrench_guided.py` | BiDex-style region/anchor ranking bằng preliminary wrench feasibility |
| IK | `solvers/fixed_contact_dls.py` | Batched DLS tới exact contact + normal |
| IK | `solvers/region_dls.py` | Batched point-to-region DLS, materialize exact contact sau solve |
| Static | `certifiers/contact_force.py` | EquiDex-style local force representation và Coulomb-cone projection |
| Static | `certifiers/grasp_wrench.py` | Grasp map/QP certificate độc lập với proposal ranking |
| Observe | `observers/contact_load.py` | MuJoCo contact → per-finger `N`, `N·m`, aggregate wrench |
| Dynamic | `validators/mujoco_rollout.py` | SynManDex-style squeeze/lift/disturbance admission |
| Control | `orchestrator.py` | Gọi đúng stage, reason accounting, chunking và artifact record |

`sample.py`, `ik.py` và `labeling.py` được giữ làm compatibility facade trong
P3.1; chúng dispatch qua allowlist strategy, không nhận import path tùy ý từ
YAML. Mỗi module nhận `numpy.random.Generator`/config tường minh, không đọc RNG
global, CWD hay mutable singleton. Static certifier không được tái sử dụng score
nội bộ của `wrench_guided` làm certificate; hai bên có thể dùng chung convention
và primitive đã test, nhưng phải chạy phép kiểm cuối độc lập trên achieved
contacts sau IK.

Ba recipe đầu tiên được hỗ trợ và ablate; chưa cho phép mọi tích tổ hợp:

| Recipe | Proposal | IK | Static + dynamic | Vai trò |
| --- | --- | --- | --- | --- |
| `surface_fixed_v1` | `surface_fixed` | `fixed_contact_dls` | chung | Correctness reference đơn giản, bắt buộc pass |
| `region_opposition_v1` | `region_opposition` | `region_dls` | chung | Đo lợi ích của target region và opposition |
| `wrench_guided_v1` | `wrench_guided` | `region_dls` | chung | Đo lợi ích của ranking theo wrench trước IK |

Force/contact tuple của EquiDexFlow, per-finger load của Transferring Contact và
staged admission của SynManDex là hạ tầng dùng chung, không phải ba recipe cạnh
tranh. Cách phân loại này tránh so sánh sai giữa một proposal method, một force
representation và một rollout protocol.

Mỗi recipe chạy cùng object list, robot list, seed set, proposal budget, IK
iteration budget và physics protocol. Báo cáo ablation phải có proposal yield,
IK convergence, static pass, dynamic pass, penetration, contact/wrench diversity,
runtime và rejection breakdown. Không chọn winner chỉ theo dynamic success vì
module có thể “thắng” bằng cách sinh nhiều grasp gần như giống nhau.

### 4.1 Surface sampler và contact set

Không gọi RNG ẩn của `trimesh.sample.sample_surface` trên đường sinh artifact.
Triển khai sampler tam giác area-weighted bằng `numpy.random.Generator` đã được
truyền vào:

1. Chọn face theo cumulative triangle area với thứ tự face ổn định.
2. Sinh barycentric coordinate bằng phép biến đổi căn bậc hai.
3. Tính point trực tiếp từ ba vertex và normal từ face đã chọn.
4. Trả cả `target_contacts [K,3]` và `target_normals [K,3]`.

Mỗi contact phải nằm trên đúng triangle đã chọn trong tolerance đã pin. Jitter,
nếu cần, chỉ được thực hiện trong barycentric/tangent coordinates rồi project
lại cùng surface; không cộng Gaussian 3D vào world point.

`surface_fixed` chọn exact contact trực tiếp và là reference tối giản.
`region_opposition` chọn surface region/anchor có opposition trước rồi giao cho
`region_dls` chọn exact reachable contact. `wrench_guided` thêm ranking bằng
object scale, separation, normal diversity/opposition và grasp-wrench feasibility
sơ bộ. Cả ba không được sao chép một patch duy nhất cho mọi ngón, chọn contact
độc lập rồi hy vọng IK tự sửa, hoặc fallback về centroid. Candidate không đạt
sau số retry hữu hạn phải bị reject có reason code.

`ContactProposal` mang exact point hoặc region membership, `target_normals`,
`target_finger_links` và force-cone/force hint tương ứng. Force hint chỉ là prior;
`contact_force`/`grasp_wrench` phải giải và certify lại trên exact achieved
contact. Identity giữa contact, normal, force và finger không được mất khi đi qua
IK, filter và physics labeling.

Palm approach tiếp tục hướng vào `-surface_normal`; palm nằm phía ngoài surface
theo standoff. Rotation phải orthonormal, `det(R)=+1`, và approach alignment phải
được test bằng dot product.

### 4.2 Batched IK và objective

`fixed_contact_dls` tạo API lõi `solve_dls_ik_batch` nhận:

- `palm_pos [B,3]`, `palm_rot [B,3,3]`;
- `target_contacts [B,K,3]`, `target_normals [B,K,3]`;
- `init_q [B,J]` tùy chọn;
- trả `q [B,J]`, `converged [B]`, per-candidate position/normal errors và số vòng
  lặp.

Wrapper single-candidate được phép giữ để tương thích, nhưng generator phải gọi
batch API theo chunk; không loop Python từng candidate trên đường chính.

`region_dls` nhận thêm triangle/point membership và mask của từng target region.
Position term là point-to-region distance khả vi hoặc nearest-point update có
step control. Khi hội tụ, solver phải materialize exact contact/face/normal và
đưa kết quả đó qua certifier; target region ban đầu không được dùng như contact
thực tế. Hai solver dùng chung joint-limit, damping và diagnostic primitive để
so sánh không bị lệch implementation.

Objective tối thiểu gồm:

- fingertip position loss theo từng tip, chuẩn hóa theo `K`;
- normal alignment giữa fingertip contact direction/frame và inverse object
  normal;
- regularization quanh `init_q`/rest pose;
- soft joint-margin penalty để nghiệm không nằm sát actuator limit nếu vẫn có
  nghiệm khả thi;
- projection joint limits sau mỗi update.

Jacobian dùng PyTorch autograd/`torch.func` trên batch và được gọi đúng là
autodiff Jacobian, không ghi “analytical Jacobian”. Có test so với central finite
difference trên fixture nhỏ. Solver cần adaptive damping hoặc step rejection khi
residual tăng; lỗi tuyến tính không được nuốt thành zero update.

Generator phải reject `converged=False` trước collision filter và MuJoCo. Manifest
ghi số candidate bị reject theo reason để audit sampler/IK coverage.

### 4.3 Physics labeling

Giữ kiến trúc mocap body + dynamic hand root + weld equality. Static
certification chạy trước dynamic rollout và hai kết quả phải được lưu riêng.
P3.1 phải:

- fail closed nếu không tạo được free root/mocap/weld; bỏ fallback mutate
  `model.body_pos` trong runtime path;
- khởi tạo đồng bộ `mocap_pos`, `mocap_quat`, free-joint pose và relative weld
  pose trước `mj_forward`;
- dùng trajectory lift tuyệt đối có velocity/acceleration hữu hạn (ví dụ
  smoothstep), không cộng offset tích lũy phụ thuộc số step;
- định nghĩa perturbation rõ là bounded base motion hoặc external wrench. Gate
  chính dùng wrench đã pin trên object qua `xfrc_applied`, ghi vector, duration
  và impulse vào metrics;
- đọc lực contact bằng `mj_contactForce`, tách normal/tangential force, và chỉ
  tính hand-object contacts thuộc allowlist contact links;
- quy đổi contact về per-finger load descriptor theo cùng đơn vị vật lý, giữ
  finger/link identity để so sánh ba hand thay vì chỉ cộng global force;
- tính penetration lớn nhất trên toàn squeeze/lift/perturbation trajectory;
- phân biệt contact với palm/support và contact từ ít nhất hai ngón độc lập.

Không được gọi “force closure” chỉ từ contact count. Chọn một trong hai cách và
ghi rõ trong schema:

1. Tính friction-cone grasp wrench matrix cùng epsilon/Q1 threshold tất định; hoặc
2. Đổi tên metric thành dynamic disturbance stability và không claim analytical
   force closure.

Phương án 1 là mặc định cho P3.1 vì plan P3 gốc đã yêu cầu force closure. Success
cuối cùng phải kết hợp: force metric đạt ngưỡng, lift/follow đạt ngưỡng,
perturbation survival, penetration dưới ngưỡng và không floor support.

Schema không được suy `dynamic_valid=True` từ static certificate. Nó lưu tối
thiểu `proposal_valid`, `ik_valid`, `collision_valid`, `static_force_valid`,
`dynamic_valid`, `failure_stage` và `failure_reason`; trường `success` chỉ là
phép hội cuối có version. Force hint do proposal/certifier tính và force đo bằng
`mj_contactForce` là hai đại lượng khác nhau, phải có tên và provenance khác nhau.

### 4.4 Dataset và provenance

`DGN-Open-Tiny` hiện tại bị coi là invalidated candidate artifact vì toàn bộ 144
sample là negative và file không được Git track. Không xóa artifact trước khi
revision record ghi hash hiện tại; regenerate vào staging directory mới rồi thay
artifact qua bước review có thể khôi phục.

`.gitignore` phải mở đúng `datasets/dgn-open-tiny/**` sau cả rule `/datasets/` và
`*.pt`, vẫn ignore mọi dataset/runtime output khác. Không dùng `git add -f`.

`scripts/check_dataset_manifest.py` phải fail nếu:

- manifest, object artifact hoặc shard phát hành không được `git ls-files` nhận;
- file vẫn khớp một ignore rule;
- shard hash sai hoặc có file ngoài manifest;
- một robot/split không có cả positive và negative sample;
- tổng positive bằng 0;
- generator commit/config/profile/object hash không đủ trong manifest.

Mỗi sample và manifest phải ghi `recipe_id`, proposal module, solver module,
certifier version và dynamic protocol version. Release có thể dùng một recipe
hoặc mixture đã chốt sau ablation, nhưng quyết định/mix weight phải nằm trong
revision record; không được âm thầm đổi strategy rồi giữ nguyên dataset version.

Hai lần regenerate từ clean staging directories với cùng seed/config phải tạo
cùng danh sách relative path và SHA-256. Gate không chấp nhận chỉ chạy lại trên
thư mục output cũ.

## 5. Work breakdown

| ID | Hạng mục | Output chính | Phụ thuộc |
| --- | --- | --- | --- |
| P3.1-00 | Khóa literature snapshot, ghi revision mở, hash dataset hiện tại và đánh dấu claim P3 bị ảnh hưởng | plan này, `docs/revisions/REV-...-p3-data-correctness.md` | không |
| P3.1-01 | Khóa typed contracts, allowlist registry, recipe schema và viết failing tests | `pipeline/contracts.py`, config, tests | P3.1-00 |
| P3.1-02 | Module `surface_fixed` deterministic | `pipeline/proposals/surface_fixed.py` | P3.1-01 |
| P3.1-03 | Module `region_opposition` scale-normalized | `pipeline/proposals/region_opposition.py` | P3.1-01/02 |
| P3.1-04 | Module `wrench_guided` và preliminary ranking | `pipeline/proposals/wrench_guided.py` | P3.1-01/03 |
| P3.1-05 | Module batched `fixed_contact_dls` | `pipeline/solvers/fixed_contact_dls.py` | P3.1-01/02 |
| P3.1-06 | Module batched `region_dls` | `pipeline/solvers/region_dls.py` | P3.1-03/05 |
| P3.1-07 | Orchestrator, compatibility facades, chunking và reason accounting | `pipeline/orchestrator.py`, `sample.py`, `ik.py`, generator | P3.1-04/05/06 |
| P3.1-08 | Harden mocap-weld initialization và smooth lift trajectory | dynamic validator | P3.1-01 |
| P3.1-09 | Contact-force observer, cone projection và grasp-wrench/QP certifier | observer/certifier modules | P3.1-01 |
| P3.1-10 | Dynamic disturbance validator và staged outcome | `validators/mujoco_rollout.py`, `labeling.py` | P3.1-08/09 |
| P3.1-11 | Known-positive/negative fixtures cho ba hand và ba recipe | tests, fixtures | P3.1-07/10 |
| P3.1-12 | Narrow Git allowlist, recipe provenance và release gate | `.gitignore`, manifest checker, tests | P3.1-01 |
| P3.1-13 | Chạy controlled ablation, chốt recipe hoặc mixture bằng revision evidence | ablation report | P3.1-11 |
| P3.1-14 | Regenerate `DGN-Open-Tiny` vào staging, audit class/recipe coverage/hash | `datasets/dgn-open-tiny/` | P3.1-12/13 |
| P3.1-15 | Chạy regression, cập nhật session/revision/roadmap bằng record mới | docs, gate evidence | P3.1-14 |

Thứ tự bắt buộc: `00 → 01 → (02 → 03 → 04) || (05 → 06) || (08 → 09 → 10)
→ 07 → 11 → (12 || 13) → 14 → 15`. Không regenerate release trước khi cả ba
recipe và static/dynamic validator pass correctness fixture.

### Trạng thái triển khai tại 2026-08-23

| Module | Trạng thái | Evidence / blocker |
| --- | --- | --- |
| P3.1-00…10 | implemented, in review | Typed contracts, surface/normal-aware IK, static GWS+gravity certificate, actuator+mocap rollout và measured contact wrench có unit/integration tests |
| P3.1-11 | blocked | LEAP và Allegro known-positive pass. Audit chứng minh Shadow phải là 24 joint state/20 actuator, không phải 20 joint + 4 mimic giả: fixed tendon truyền tổng `J1+J2` nhưng không ràng buộc `J1=J2`. Profile Shadow hiện `release_blocked`; validator fail `underactuated_targets` cho tới khi có underactuated command/control solver. Không được hạ physics gate để làm fixture xanh |
| P3.1-12 | implemented, in review | Manifest v2, source/profile hashes, clean-commit flag, exact release-file audit và Git tracked/non-ignored checks đã có test |
| P3.1-13 | pending rerun | Kết quả cũ trong `REV-20260823-002` bị invalidated vì dùng pipeline trước correctness fixes. Script mặc định dry-run, cần `--execute`, giới hạn tối đa 96 candidates và rate dùng mẫu số theo từng stage |
| P3.1-14…15 | pending | Không regenerate hoặc đóng P3.1 trước khi P3.1-11 và controlled ablation pass |

Do máy development từng bị hard-freeze trong ablation, mọi verification còn lại
phải chạy tuần tự với `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1` và timeout hữu hạn. Không chạy tìm kiếm pose hoặc
ablation không giới hạn để đổi lấy một fixture dương.

## 6. Test bắt buộc

### Surface/contact tests

- Cả ba proposal module thỏa cùng `ContactProposal` contract và không import/gọi
  solver, certifier hoặc MuJoCo.
- Point thỏa barycentric triangle và khoảng cách tới surface `<= tolerance`.
- Normal unit-length, finite và đúng hướng face.
- Contact/normal/force/finger identity giữ nguyên qua sampler → IK → labeling.
- Force proposal nằm trong Coulomb cone hoặc bị reject trước simulation.
- Palm ở ngoài surface; approach alignment đúng; rotation thuộc `SO(3)`.
- Cùng seed cho byte-equivalent candidates; seed khác tạo diversity.
- Không contact set nào vi phạm minimum separation đã scale theo object.
- Monkeypatch `trimesh.sample.sample_surface` để ném lỗi nhưng sampler mới vẫn
  chạy, chứng minh không còn RNG ẩn.
- `region_opposition` trả region membership hợp lệ cho từng finger;
  `wrench_guided` ranking thay đổi có kiểm soát khi normal/opposition thay đổi.

### IK tests

- Autodiff Jacobian gần central finite difference trong tolerance.
- Batch result gần single wrapper cho cùng input.
- Joint limits, singular configuration, unreachable target và mixed
  converged/non-converged batch.
- Fixture reachable cho LEAP, Allegro, Shadow hội tụ theo per-tip position và
  normal thresholds.
- `region_dls` chọn exact post-IK contact nằm trong target region, trên đúng
  surface, và không tái sử dụng stale normal của proposal.
- Generator không gọi physics cho nghiệm `converged=False`.

### Physics tests

- No-contact, palm-only contact, floor-supported object và excessive penetration
  đều fail.
- Known stable grasp của mỗi hand pass lift + wrench perturbation.
- Force metric bằng 0 khi không có lực dù contact record hình học tồn tại.
- Per-finger load cộng lại khớp total hand-object contact wrench trong tolerance.
- Penetration spike giữa trajectory làm sample fail dù frame cuối sạch.
- Lift path độc lập với số step trong tolerance vật lý đã pin.
- Mocap pose/quaternion và weld không tạo jump ở frame đầu.
- Static-pass/dynamic-fail fixture giữ hai status khác nhau và ghi đúng
  `failure_stage`; predicted force hint không bị ghi đè lên measured force.

### Module/recipe tests

- Registry chỉ nhận ba recipe/version đã allowlist; import path tùy ý và tổ hợp
  chưa duyệt phải fail closed.
- Compatibility facade cho API P3 cũ cho kết quả tương đương
  `surface_fixed_v1` trên fixture đã pin.
- Mỗi module deterministic theo RNG/config truyền vào và không phụ thuộc thứ tự
  chạy recipe.
- Cả ba recipe chạy end-to-end trên cùng fixture cho mỗi hand, có đầy đủ stage
  evidence kể cả khi bị reject.
- Controlled ablation dùng cùng seed/object/budget và báo conversion rate từng
  stage, diversity lẫn runtime; checker fail nếu thiếu recipe hoặc budget lệch.

### Dataset/release tests

- Mỗi `(split, robot)` có positive và negative samples.
- Sample ghi `recipe_id` và version của proposal/solver/certifier/validator;
  manifest counts theo recipe khớp shard.
- Manifest counts khớp nội dung shard thực, không chỉ metadata khai báo.
- Hai clean regenerations cùng seed cho cùng relative paths và SHA-256.
- `git check-ignore` trả non-match và `git ls-files` trả match cho mọi artifact
  phát hành.
- Gate fail với all-negative dataset, untracked manifest, force-added shard,
  stale file, profile hash drift và generator commit drift.

## 7. Acceptance thresholds

Threshold số phải nằm trong config/manifest, không hard-code rải rác. Giá trị cụ
thể được chốt ở P3.1-01 bằng fixture đo được, nhưng các invariant sau không được
hạ thấp:

- Surface distance: theo mesh scale/tolerance, tối đa không quá `1e-6 m` với
  primitive test chính xác.
- Rotation: orthonormal và determinant dương trong tolerance FP64.
- IK: báo convergence theo max per-tip position error và normal angular error;
  không dùng một scalar tổng che mất fingertip xấu.
- Physics: ít nhất hai finger groups có normal force dương; palm-only không đạt.
- Dataset: cả hai class xuất hiện trong từng robot và split; không shard
  all-negative/all-positive.
- Reproducibility: danh sách file và SHA-256 giống hệt giữa hai clean runs.
- Provenance: 100% artifact release được Git track bằng allowlist thường.
- Modularity: cả ba recipe pass contract/determinism fixture; không module nào
  được bypass static hoặc dynamic stage.

Không đặt success-rate mục tiêu theo paper SOTA cho tiny procedural fixture;
P3.1 đo tính hợp lệ của pipeline, không tuyên bố benchmark tương đương paper.

## 8. Gate scripts cần sửa

`scripts/check_phase3.py` phải bổ sung ít nhất:

1. Registry/contract và surface/contact invariants cho ba recipe, ba hand.
2. Batched fixed/region IK convergence và explicit non-convergence rejection.
3. Static-certificate và known-positive/known-negative dynamic fixtures.
4. Staged reason accounting và controlled-ablation budget check.
5. Dataset class/recipe coverage theo robot/split.
6. Clean-regeneration hash comparison hoặc gọi checker chuyên trách.
7. Git tracked/non-ignored release verification.

`scripts/check_dataset_manifest.py` vẫn stdlib-only ở phần manifest/Git. Test
geometry/IK/physics nằm trong `check_phase3.py`, không kéo NumPy/MuJoCo vào
manifest checker.

## 9. Gate đóng P3.1

```bash
.venv/bin/python scripts/check_environment.py --profile cpu
.venv/bin/python scripts/check_references.py --lock-only
.venv/bin/python scripts/check_references.py --source-root .references
.venv/bin/python scripts/check_train_args.py --registry-only
.venv/bin/python scripts/check_train_args.py --source .references/ultralytics
.venv/bin/python scripts/check_docs.py --root .
.venv/bin/python -m unittest discover -s scripts/tests -p 'test_*.py'
.venv/bin/python scripts/check_phase0.py
.venv/bin/python scripts/check_phase1.py
.venv/bin/python scripts/check_robot_assets.py
.venv/bin/python scripts/check_phase2.py
.venv/bin/python scripts/check_dataset_manifest.py --root datasets/dgn-open-tiny
.venv/bin/python scripts/check_phase3.py
.venv/bin/python -m pytest tests/ -q
git diff --check
git status --short
```

Ngoài ra revision record phải lưu:

- hash trước/sau của sampler, IK, labeling, generator, checker, `.gitignore`,
  manifest và sáu shard;
- class counts trước/sau theo robot/split;
- kết quả hai clean regenerations;
- benchmark batch IK ở batch size đã pin, chỉ dùng làm evidence throughput,
  không thay correctness gate.
- ablation table của ba recipe, config/budget hash và lý do chọn recipe/mixture
  cho release;
- literature cutoff và danh sách nguồn primary được kiểm lại nếu implementation
  bắt đầu sau `2026-09-23`; nguồn mới không tự động đổi scope mà phải qua revision.

## 10. Điều kiện chuyển P4

Chỉ mở P4 khi:

1. Tất cả gate §9 pass trên clean worktree và CPU lock.
2. `DGN-Open-Tiny` mới có positive/negative coverage cho cả ba hand và hai split.
3. Artifact được Git track, manifest hash đúng và regenerate bit-identical.
4. Cả ba recipe pass module gate; recipe/mixture phát hành đã được chốt từ
   controlled ablation và ghi vào manifest.
5. Revision record sửa `SESSION-20260822-022` đã liệt kê rõ claim nào được thay
   thế; không sửa trực tiếp session cũ.
6. `PROJECT_PHASES.md` chỉ chuyển P3 sang `complete` sau các điều kiện trên.
7. Independent reviewer xác nhận ít nhất sampler/IK fixture và physics labeling;
   nếu chưa có reviewer, trạng thái phải là `in_review` và P4 tiếp tục bị chặn.
