---
document_id: TRAIN-ARGS-ULTRALYTICS-8.4.125
document_type: registry
title: Registry đầy đủ tham số train tương thích Ultralytics v8.4.125
status: active
date: 2026-08-21
revises: none
related_plan: PLAN-V2
source_commit: 329682a29d27203582ba30e519340f95abccc6a6
source_sha256: eb5e9ab6825a5d55076f8b38aed00953dec722ed5d5368a6584df35f50f32839
---

# Registry tham số train Ultralytics → DexGrasp

Đây là bảng kiểm kê thiết kế, chưa phải tuyên bố runtime DexGrasp đã được cài
đặt. Nguồn sự thật máy đọc được là `TRAIN_ARGUMENTS.yaml`; file này giải thích
rõ từng nhóm để không một tham số upstream nào bị bỏ qua hoặc trở thành no-op
im lặng.

## 1. Nguồn và mức bao phủ

| Thành phần | Giá trị đã khóa |
|---|---|
| Repository | `https://github.com/ultralytics/ultralytics.git` |
| Tag / commit | `v8.4.125` / `329682a29d27203582ba30e519340f95abccc6a6` |
| Config | `ultralytics/cfg/default.yaml` |
| SHA-256 config | `eb5e9ab6825a5d55076f8b38aed00953dec722ed5d5368a6584df35f50f32839` |
| Canonical YAML keys | 115; zero missing, zero extra |
| Extra config kwargs | 2: `augmentations`, `save_dir` |
| Legacy names | 9: 6 alias và 3 tên nhận rồi loại |
| API control | 1: `trainer` trong `Model.train(trainer=None, **kwargs)` |
| Tổng bề mặt tên | 127 |

Ultralytics dùng một config chung cho train/val/predict/export/track. Vì vậy cả
115 key đều qua `get_cfg()` khi dựng trainer, nhưng “được nhận” không đồng nghĩa
“có tác dụng trong train”. Registry vẫn ghi tất cả key ngoài train để tránh hiểu
nhầm một key bị bỏ sót với một key đã được quyết định là không áp dụng.

Disposition:

- `R` (`retain`): giữ tên và nghĩa public.
- `A` (`adapt`): giữ tên nhưng có hợp đồng grasp/device rõ dưới đây.
- `D` (`defer`): đã dành tên cho mode sau; v1 phải báo chưa hỗ trợ nếu người dùng
  đặt nó, không bỏ qua.
- `X` (`reject`): key đã biết nhưng không có nghĩa đúng cho grasp; truyền tường
  minh phải lỗi kèm hướng dẫn key thay thế nếu có.

Tổng hiện tại: 31 `R`, 27 `A`, 8 `D`, 49 `X`. Đây là quyết định thiết kế có thể
được sửa qua revision record; không phải 49 key bị quên.

## 2. Thứ tự merge, validation và resume

Thứ tự upstream được giữ làm contract tham chiếu:

```text
get_cfg: default config → overrides
Model.train: model overrides → method defaults → kwargs → mode=train bắt buộc
```

Unknown key phải lỗi và gợi ý tên gần nhất. Không cho phép dead key/no-op. Trong
115 key, upstream kiểm kiểu tập trung cho 93 key: 16 số, 26 fraction, 11 integer,
33 boolean, 4 string và ba luật riêng cho `scale`, `compile`, `quantize`. Còn 22
key không được `check_cfg()` kiểm kiểu tập trung và phải qua validator downstream:

`amp`, `cache`, `cfg`, `classes`, `data`, `device`, `distill_model`, `embed`,
`format`, `freeze`, `imgsz`, `mode`, `model`, `name`, `opset`, `pretrained`,
`project`, `resume`, `source`, `stream_buffer`, `task`, `tracker`.

Các key upstream cho phép đổi khi exact-resume gồm `imgsz`, `batch`, `device`,
`close_mosaic`, `augmentations`, `save_period`, `workers`, `cache`, `patience`,
`time`, `freeze`, `val`, `plots`, `distill_model`, `save_dir`. DexGrasp chỉ được
giữ subset có disposition `R/A`; mọi thay đổi robot schema, model YAML, optimizer,
scheduler hoặc dataset manifest phải bị từ chối trước khi load state.

## 3. Global và train settings — 35 key

| Key | Default | Kiểu upstream/documented | Dex | Hợp đồng DexGrasp |
|---|---:|---|:---:|---|
| `task` | `detect` | enum | A | Giá trị hợp lệ của package là `grasp`; model quyết định task. |
| `mode` | `train` | enum | R | `Model.train()` luôn ép `train`. |
| `model` | `null` | path/null | R | Model YAML hoặc safe checkpoint `.pt`. |
| `data` | `null` | path/null | R | Dataset YAML/manifest; bắt buộc trừ exact resume. |
| `epochs` | `100` | integer | R | Budget epoch; `time`/`max_steps` có thể dừng sớm hơn. |
| `time` | `null` | number/null | R | Giờ tối đa; nếu đặt thì thắng `epochs` về điều kiện dừng. |
| `patience` | `100` | integer | R | Early stop theo metric đã định danh, không theo total loss tùy ý. |
| `batch` | `16` | number | A | Integer cho CPU/CUDA; fraction chỉ AutoBatch CUDA đơn. |
| `imgsz` | `640` | int/list | A | Chỉ là resize đầu vào depth; point-cloud trực tiếp không dùng và phải cảnh báo/lỗi nếu override vô nghĩa. |
| `save` | `True` | boolean | R | Lưu last/best và run artifacts. |
| `save_period` | `-1` | integer | R | `<1` tắt periodic checkpoint; `N≥1` lưu mỗi N epoch. |
| `cache` | `False` | bool/enum | A | `False`, `ram`, `disk`; kiểm RAM/disk, deterministic và ghi effective mode. |
| `device` | `null` | selector | A | v1 hỗ trợ CPU và NVIDIA CUDA; MPS/XPU/NPU là giá trị upstream đã biết nhưng ngoài phạm vi v1. |
| `workers` | `8` | integer | A | CPU có safe effective value; luôn ghi requested/effective, không đổi im lặng. |
| `project` | `null` | path/null | R | Root của run. |
| `name` | `null` | string/null | R | Tên experiment. |
| `exist_ok` | `False` | boolean | R | Cho phép dùng run directory đã tồn tại. |
| `pretrained` | `True` | bool/path | A | Chỉ load weights có model YAML, robot/profile và joint schema tương thích. |
| `cls_remap` | `True` | boolean | X | Remap class head YOLO không tương đương remap joint/robot. |
| `optimizer` | `auto` | string | R | `auto` hoặc optimizer được whitelist; effective choice/LR phải được log. |
| `verbose` | `True` | boolean | R | Mức log public. |
| `seed` | `0` | integer ≥0 | R | Seed Python/NumPy/Torch/dataloader/simulator. |
| `deterministic` | `True` | boolean | R | Bật deterministic ops nếu backend hỗ trợ; incompatibility phải báo rõ. |
| `single_cls` | `False` | boolean | X | Không có class collapse trong task grasp. |
| `rect` | `False` | boolean | X | Rectangular image batching không áp dụng cho point tokens. |
| `cos_lr` | `False` | boolean | R | Cosine scheduler thay linear scheduler. |
| `close_mosaic` | `10` | integer | X | Mosaic ảnh không tồn tại; không dùng nó để tắt augmentation 3D. |
| `resume` | `False` | bool/path | A | Exact state resume: optimizer/scheduler/scaler/RNG/manifest/profile. |
| `amp` | `True` | boolean | A | CUDA capability check; CPU luôn effective `False` và ghi vào run config. |
| `fraction` | `1.0` | `(0,1]` | R | Fraction dataset dùng để train. |
| `profile` | `False` | boolean | A | Chỉ profile PyTorch paths; không tự kích hoạt TensorRT/ONNX. |
| `freeze` | `null` | int/list/null | R | Freeze N layer đầu hoặc index; lỗi nếu không còn parameter trainable. |
| `multi_scale` | `0.0` | `[0,1]` | X | Multi-scale ảnh không được áp sang point cloud bằng cùng tên. |
| `compile` | `False` | bool/string | A | CPU/CUDA có preflight; failure phải ghi fallback hoặc fail trước batch đầu. |
| `channels_last` | `False` | boolean | X | Kiến trúc point dùng tensor layout khác NCHW/NHWC. |

## 4. Task-specific và validation — 13 key

| Key | Default | Kiểu | Dex | Hợp đồng DexGrasp |
|---|---:|---|:---:|---|
| `overlap_mask` | `True` | boolean | X | Mask instance YOLO không thuộc nhãn grasp. |
| `mask_ratio` | `4` | integer ≥1 | X | Không downsample segmentation mask. |
| `dropout` | `0.0` | `[0,1]` | A | Override dropout chỉ cho module YAML khai báo hỗ trợ. |
| `val` | `True` | boolean | R | Chạy validator trong training. |
| `split` | `val` | string | R | Tên split từ dataset manifest. |
| `save_json` | `False` | boolean | A | Lưu `GraspResults` JSON cùng schema/version/hash, không phải COCO JSON. |
| `conf` | `null` | `[0,1]`/null | A | Ngưỡng score grasp cho val/predict; không đổi loss. |
| `iou` | `0.7` | `[0,1]` | X | Box NMS IoU; dùng extension `radius` cho seed suppression. |
| `max_det` | `300` | integer ≥1 | A | Số grasp tối đa mỗi sample trước/sau validation postprocess được ghi rõ. |
| `quantize` | `null` | precision/null | A | Val/export precision; không điều khiển train AMP. CPU mặc định FP32. |
| `dnn` | `False` | boolean | X | OpenCV DNN không thuộc backend v1. |
| `plots` | `True` | boolean | A | Plot loss/metric và grasp diagnostic có schema. |
| `end2end` | `null` | bool/null | X | YOLO end-to-end detection head không áp dụng. |

## 5. Predict và visualize — 18 key

Các key này nằm trong config chung nhưng không điều khiển optimizer/train loop.

| Key | Default | Kiểu | Dex | Hợp đồng DexGrasp |
|---|---:|---|:---:|---|
| `source` | `null` | source/null | A | Point cloud, depth+intrinsics hoặc path/iterator được hỗ trợ. |
| `vid_stride` | `1` | integer ≥1 | D | Chỉ khi có depth/video streaming contract. |
| `stream_buffer` | `False` | boolean | D | Dành cho predictor streaming. |
| `visualize` | `False` | boolean | A | Xuất point/token/grasp diagnostics; không đổi prediction. |
| `augment` | `False` | boolean | D | TTA 3D phải có merge/frame contract trước khi bật. |
| `agnostic_nms` | `False` | boolean | X | Class-agnostic box NMS không tồn tại. |
| `classes` | `null` | int/list/null | X | Không lọc class; robot/object filters cần key riêng. |
| `retina_masks` | `False` | boolean | X | High-resolution segmentation masks không tồn tại. |
| `embed` | `null` | list/null | D | Dành cho API trích feature sau khi khóa layer naming. |
| `show` | `False` | boolean | A | Hiển thị 3D grasp/point cloud khi môi trường có renderer. |
| `save_frames` | `False` | boolean | D | Chỉ có nghĩa với streaming. |
| `save_txt` | `False` | boolean | A | Text schema grasp versioned; không dùng YOLO xywh. |
| `save_conf` | `False` | boolean | A | Ghi score breakdown cùng prediction artifact. |
| `save_crop` | `False` | boolean | X | Image crop không đại diện local geometry token. |
| `show_labels` | `True` | boolean | A | Hiện robot/joint/object identifiers trong plot. |
| `show_conf` | `True` | boolean | A | Hiện grasp score. |
| `show_boxes` | `True` | boolean | X | Không vẽ bounding box như một grasp pose. |
| `line_width` | `null` | int/null | A | Độ dày line/axes trong plot 2D/3D. |

## 6. Export, custom và tracker — 10 key

| Key | Default | Kiểu | Dex | Hợp đồng DexGrasp |
|---|---:|---|:---:|---|
| `format` | `torchscript` | string | A | v1 whitelist TorchScript/ONNX; format khác lỗi rõ. |
| `keras` | `False` | boolean | X | TensorFlow/Keras ngoài phạm vi v1. |
| `optimize` | `False` | boolean | X | Cờ DEEPX upstream ngoài phạm vi. |
| `dynamic` | `False` | boolean | R | Dynamic batch/point axes nếu backend bundle hỗ trợ. |
| `simplify` | `True` | boolean | A | Chỉ ONNX và chỉ khi dependency/version đã pin. |
| `opset` | `null` | int/null | R | ONNX opset tường minh hoặc default exporter đã pin. |
| `workspace` | `null` | number/null | X | TensorRT workspace ngoài phạm vi v1. |
| `nms` | `False` | boolean | X | Không nhúng box NMS; grasp suppression có contract riêng. |
| `cfg` | `null` | path/null | R | YAML override entry point; unknown/dead key là lỗi. |
| `tracker` | `tracktrack.yaml` | path/string | X | Tracking ngoài phạm vi v1. |

## 7. Hyperparameters — 39 key

| Key | Default | Kiểu | Dex | Hợp đồng DexGrasp |
|---|---:|---|:---:|---|
| `lr0` | `0.01` | `[0,1]` | R | Initial LR; `optimizer=auto` phải log nếu override nó. |
| `lrf` | `0.01` | `[0,1]` | R | Final LR = `lr0*lrf`. |
| `momentum` | `0.937` | `[0,1]` | R | SGD momentum/Adam beta1. |
| `weight_decay` | `0.0005` | `[0,1]` | R | Scaled theo effective batch/accumulation như contract trainer. |
| `warmup_epochs` | `3.0` | number | R | Warmup epoch, cho phép fraction. |
| `warmup_momentum` | `0.8` | `[0,1]` | R | Momentum đầu warmup. |
| `warmup_bias_lr` | `0.1` | `[0,1]` | R | Bias LR đầu warmup. |
| `distill_model` | `null` | path/null | D | Chưa có distillation contract cho grasp/robot schema. |
| `dis` | `6.0` | number | D | Loss weight dành cùng feature distillation. |
| `box` | `7.5` | number | X | Dùng `loss_*` extension, không đổi nghĩa box thành wrist pose. |
| `cls` | `0.5` | number | X | Không có YOLO classification loss. |
| `cls_pw` | `0.0` | `[0,1]` | X | Không có class-frequency power contract. |
| `dfl` | `1.5` | number | X | Không có distribution focal box loss. |
| `pose` | `12.0` | number | X | Key YOLO keypoint pose; không tái dùng cho SE(3) wrist loss. |
| `kobj` | `1.0` | number | X | Keypoint objectness không tồn tại. |
| `rle` | `1.0` | number | X | YOLO pose RLE loss không thuộc grasp head. |
| `angle` | `1.0` | number | X | OBB angle loss không thay rotation-on-SO(3) loss. |
| `dlog` | `1.0` | number | X | Depth task loss không thuộc grasp objective. |
| `dgrad` | `0.5` | number | X | Depth gradient loss không thuộc grasp objective. |
| `dlam` | `1.0` | `[0,1]` | X | Depth SILog focus không thuộc grasp objective. |
| `nbs` | `64` | integer ≥1 | R | Nominal batch cho accumulation/loss normalization. |
| `hsv_h` | `0.015` | `[0,1]` | X | Input learned v1 là XYZ; không HSV augmentation. |
| `hsv_s` | `0.7` | `[0,1]` | X | Như trên. |
| `hsv_v` | `0.4` | `[0,1]` | X | Như trên. |
| `degrees` | `0.0` | number | X | Rotation ảnh 2D; dùng `rotation_aug_deg` với frame/SE(3). |
| `translate` | `0.1` | `[0,1]` | X | Fraction ảnh 2D; dùng `translation_aug_m`. |
| `scale` | `0.5` | number/pair | X | Scale ảnh không được áp vào robot geometry. |
| `shear` | `0.0` | number | X | Shear ảnh không bảo toàn rigid geometry. |
| `perspective` | `0.0` | `[0,1]` | X | Perspective warp không thay depth projection có calibration. |
| `flipud` | `0.0` | `[0,1]` | X | Flip có thể đổi handedness/frame; không dùng im lặng. |
| `fliplr` | `0.5` | `[0,1]` | X | Như trên. |
| `bgr` | `0.0` | `[0,1]` | X | Không có RGB↔BGR trong XYZ-only v1. |
| `mosaic` | `1.0` | `[0,1]` | X | Ghép ảnh không có grasp/scene collision contract. |
| `mixup` | `0.0` | `[0,1]` | D | Dành tên cho point-scene mixup sau khi có nhãn/collision rule. |
| `cutmix` | `0.0` | `[0,1]` | X | Không cắt-dán rigid scene bằng rule ảnh. |
| `copy_paste` | `0.0` | `[0,1]` | X | Cần scene composition/simulation riêng. |
| `copy_paste_mode` | `flip` | string | X | Strategy segmentation không áp dụng. |
| `auto_augment` | `randaugment` | string | X | Policy classification ảnh không áp dụng. |
| `erasing` | `0.4` | `[0,1]` | X | Dùng `point_dropout` có frame/label contract. |

## 8. Hai extra kwargs, chín legacy names và API control

| Tên | Upstream behavior | DexGrasp decision |
|---|---|---|
| `augmentations` | Whitelist ngoài `default.yaml` | A: Python API nhận callable an toàn; YAML chỉ nhận transform registry có tên. |
| `save_dir` | Whitelist runtime/internal | R: path output đã resolve; checkpoint lưu effective path. |
| `boxes` | Alias cũ → `show_boxes` | X vì target `show_boxes` bị reject. |
| `hide_labels` | Alias đảo → `show_labels` | A, cảnh báo deprecated. |
| `hide_conf` | Alias đảo → `show_conf` | A, cảnh báo deprecated. |
| `line_thickness` | Alias → `line_width` | A, cảnh báo deprecated. |
| `half` | Alias → `quantize=16` | A cho val/export; không điều khiển train AMP. |
| `int8` | Alias → `quantize=8` | A cho backend đã hỗ trợ; v1 có thể reject theo format/device. |
| `label_smoothing` | Nhận rồi loại với warning | X; không no-op im lặng. |
| `save_hybrid` | Nhận rồi loại với warning | X; không no-op im lặng. |
| `crop_fraction` | Nhận rồi loại với warning | X; không no-op im lặng. |
| `trainer` | Tham số riêng của `Model.train` | A: trainer class/instance phải tuân engine contract và callback lifecycle. |

## 9. Hợp đồng CPU/CUDA bắt buộc

| Key/hành vi | CPU | CUDA đơn | CUDA DDP |
|---|---|---|---|
| `device` | `cpu`; FP32 chuẩn | `0`, `cuda:0`, auto selector | danh sách device; spawn process |
| `batch` integer | Hỗ trợ | Hỗ trợ | Phải chia hợp lệ theo world size |
| `batch` fraction | Lỗi sớm, yêu cầu integer; không giả vờ AutoBatch | AutoBatch theo fraction VRAM | Lỗi sớm như upstream |
| `amp=True` | Effective `False`, ghi warning/config | Capability check rồi AMP/scaler | Rank 0 check rồi broadcast |
| `workers` | Safe default/effective value được ghi lại | Cap theo CPU/batch/device | Mỗi rank có effective workers |
| `compile` | Preflight PyTorch/compiler; fallback/failure được ghi trước train | Preflight backend/mode | Kiểm static graph và unused params |
| `channels_last` | Reject | Reject vì point layout, không phải vì thiếu CUDA | Reject |
| OOM recovery | Có thể giảm integer batch trong epoch đầu, tối đa ba lần và ghi event | Tương tự | Không tự đổi mỗi rank độc lập |
| cache/pin memory | Kiểm RAM/disk; không pin accelerator memory | Pin memory khi phù hợp | Giới hạn theo từng rank |

Mọi run lưu cả requested config và effective config. CPU không được đi vào CUDA
API, không được import dependency CUDA-only ở base install, và smoke train CPU
phải được chạy trong release gate runtime khi package tồn tại.

## 10. DexGrasp extensions đang chờ khóa thiết kế

Registry máy đã dành tên cho `robot`, `sim`, `max_steps`, `points_per_scene`,
`voxel_size`, `max_tokens`, `flow_steps`, `topk`, `radius`, bốn augmentation 3D,
năm loss gain và `quality_negatives`. Tất cả đang có
`status: required_design`: chưa có default/range nào được coi là đã chấp nhận.
Khi implement, feature tương ứng phải thêm config, validator, CPU/CUDA tests,
revision record và independent review trước khi chuyển sang `implemented`.

## 11. Kiểm tra chống bỏ sót

Không cần clone (hook/CI bootstrap):

```bash
python3 scripts/check_train_args.py --registry-only
```

Đối chiếu đầy đủ với source chuẩn:

```bash
python3 scripts/check_train_args.py --source .references/ultralytics
```

Full check fail khi HEAD/hash sai, canonical key/default/group sai, membership
type/range sai, thiếu/thừa custom hoặc legacy key, hay chữ ký `Model.train` đổi.
Nâng Ultralytics phải cập nhật clone và registry trong integration feature; không
được sửa count/hash chỉ để làm checker xanh.
