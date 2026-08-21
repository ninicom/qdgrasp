---
document_id: DOCS-SCHEMA-001
document_type: policy
title: Hợp đồng kiểm tra tài liệu
status: active
date: 2026-08-21
revises: none
---

# Hợp đồng kiểm tra tài liệu

Chạy từ bất kỳ thư mục nào:

```bash
python scripts/check_docs.py --root /đường/dẫn/tới/Dexgraspnet_custom
```

Chạy regression test mà không tạo bytecode trong workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s scripts/tests -p 'test_*.py' -v
```

Kiểm registry tham số train mà không cần source clone:

```bash
python3 scripts/check_train_args.py --registry-only
```

Đối chiếu registry với đúng clone Ultralytics đã pin:

```bash
python3 scripts/check_train_args.py --source .references/ultralytics
```

Validator chỉ dùng Python standard library. Phạm vi tài liệu được quản lý gồm:

- `PLAN.md`, `docs/README.md`, `docs/governance/*.md`,
  `docs/decisions/*.md`, `docs/configuration/*.md` và
  `docs/schemas/README.md`;
- index `README.md` của `archive`, `configuration`, `reports`, `reviews`,
  `revisions`, `sessions`;
- record Markdown trong `docs/reports/`, `docs/sessions/`, `docs/reviews/`,
  `docs/revisions/`, `docs/metrics/`.

Thư mục record chưa tồn tại hoặc rỗng là hợp lệ. Template tại `docs/templates/`,
file bắt đầu bằng `_`/`.` và nguyên bản bất biến
`docs/archive/PLAN.pre-v2.md` không được quét như record. Archive nguyên bản được
quản lý gián tiếp qua `docs/archive/README.md` và revision record có checksum.

Mọi tài liệu trong phạm vi phải có `document_id`, `document_type`, `title`,
`status`, `date`, `revises` trong YAML front matter. Tài liệu quy phạm (`plan`,
`policy`, `decision`, `registry`, `index`) được dùng `status: active`; record
phiên/số liệu/review/revision không được dùng `active`.

## Quy tắc theo loại tài liệu

- `session_report` dùng YAML front matter với `document_id`, `document_type`,
  `title`, `status`, `date`, `session_id`, `author`, `revises` và
  `related_plan`. Các mục H2 bắt buộc: “Mục tiêu phiên”, “Việc đã hoàn tất”,
  “Bằng chứng”, “Kiểm tra đã chạy”, “Việc chưa hoàn tất”, “Sửa đổi phiên
  trước” và “Bàn giao”.
- `third_party_review` và `revision_record` tuân theo hai template cùng tên
  trong `docs/templates/`. Validator kiểm tra metadata, enum và các mục bằng
  chứng/kết luận bắt buộc.
- Metrics Registry được nhận diện bằng tiêu đề H1 và bắt buộc có front matter
  chung với `document_type: registry`. Báo cáo số liệu dùng front matter
  `metrics_report`. Validator kiểm tra các H2, định danh protocol/môi trường,
  `n`, CI 95%, raw artifact, SHA-256, comparability, validity và reviewer.
- Báo cáo Markdown khác trong `docs/reports/` phải có front matter tối thiểu và
  các mục “Tóm tắt”, “Bằng chứng”, “Kết luận”; `release_report` dùng thêm mẫu
  `docs/templates/RELEASE_REPORT.md`.
- Markdown ngoài README trong `docs/configuration/` phải có
  `document_type: registry`. `TRAIN_ARGUMENTS.yaml` được checker riêng xác minh
  số lượng, field bắt buộc, disposition, type-set và source provenance.

Heading có thể có tiền tố số (`## 3. Bằng chứng`); validator bỏ tiền tố này khi
đối chiếu. Front matter chỉ cần scalar ở cấp cao nhất và list lồng dưới
`revises`; đây là chủ ý để công cụ bootstrap không phụ thuộc PyYAML.

Khi `status` khác `draft`, metadata không được còn token mẫu như `YYYY-MM-DD`,
`NNN`, `RUNID`, `AUTHOR-ID`, `SESSION-ID`, “Tên ngắn của phiên”,
`tên-người-thực-hiện`, dấu `<...>` hoặc `{...}`. Vì vậy việc chỉ đổi một template
từ `draft` sang `complete` sẽ bị từ chối cho tới khi metadata đã được điền thật.

## Sửa kết quả của phiên trước

`revises: none`, `revises: []` hoặc giá trị tương đương nghĩa là không sửa phiên
trước. Với mọi giá trị khác, tài liệu bắt buộc có nguyên nhân cụ thể, mức cần
thiết hợp lệ và tác động:

- session: `revision_reason`, `necessity: N0|N1|N2|N3`, `impact`;
- revision record: `reason`, `necessity: N0|N1|N2|N3`, `impact`;
- báo cáo chung cũng chỉ dùng một trong bốn mức `N0|N1|N2|N3`.

Ba trường này không được để trống hoặc dùng placeholder, kể cả khi tài liệu còn
ở trạng thái draft. Bản ghi phiên cũ phải được giữ nguyên; chỉnh sửa được ghi ở
một bản ghi mới có liên kết ngược tới phiên/artifact/revision cũ.

Vocabulary record là `draft`, `in_review`, `complete`, `accepted`, `superseded`
và `invalidated`. Tài liệu quy phạm dùng thêm `active`. Không dùng `completed`
hoặc `final`.

## Hợp đồng train-argument registry

`scripts/check_train_args.py --registry-only` khóa 115 canonical key, hai extra
config kwargs, chín legacy name và một API control. Chế độ full còn kiểm exact
Git HEAD, SHA-256 của `default.yaml`, config validator, model API, trainer, toàn
bộ default/group, type/range membership và chữ ký `Model.train`. Exit code `1`
là mismatch nội dung; `2` là invocation/source path không hợp lệ.
