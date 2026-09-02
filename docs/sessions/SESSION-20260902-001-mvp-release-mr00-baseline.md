---
document_id: SESSION-20260902-001
document_type: session_report
title: MVP release MR-00 — khóa phạm vi và baseline
status: complete
date: 2026-09-02
session_id: SESSION-20260902-001
author: codex-agent
revises: none
related_plan: ROADMAP-MVP-RELEASE-001
scope: mujoco_cpu_leap_state_cuboid_only
candidate_branch: feature/mvp-grasp-policy
candidate_commit: 90b3accb76ae1417199167661c58cd937b00d9b9
---

# SESSION-20260902-001 — MVP release MR-00: khóa phạm vi và baseline

## Mục tiêu phiên

- Khóa supported scope cho MVP release trước khi tạo artifact mới.
- Chụp baseline branch, worktree và MVP artifact gate để MR-01 có điểm bắt đầu
  có thể truy vết.

## Việc đã hoàn tất

| ID | Kết quả hoàn tất | Output | Trạng thái xác minh |
| --- | --- | --- |
| `MR00-01` | Khóa release scope: MuJoCo CPU, LEAP, privileged state và cuboid | `ROADMAP-MVP-RELEASE-001` §0 | verified |
| `MR00-02` | Loại robot thật, multi-hand, camera, clutter và GPU physics khỏi release | cùng plan | verified |
| `MR00-03` | Chụp branch/worktree/candidate baseline | branch `feature/mvp-grasp-policy`, commit `90b3acc` | verified |
| `MR00-04` | Xác nhận evidence MVP v0 không đạt contract release mới | `runs/mvp`, 50/75 | verified |

Release đầu tiên chỉ hỗ trợ MuJoCo CPU, LEAP Hand profile đã pin, privileged
state observation và một cuboid trên bàn trong MVP scope v1 sắp tạo. Robot thật,
Allegro, Shadow, raw mesh, clutter, camera observation, GPU physics và
sim-to-real không thuộc release này. Không có actuator command path công khai
được tạo trong workstream MVP release.

## Bằng chứng

| Evidence ID | Loại | Đường dẫn/URI | SHA-256 hoặc tham chiếu bất biến |
| --- | --- | --- |
| `MR00-E01` | git state | `feature/mvp-grasp-policy` | HEAD `90b3accb76ae1417199167661c58cd937b00d9b9` |
| `MR00-E02` | worktree state | `git status --short --branch` | 8 MVP file sửa đổi, 1 plan mới chưa track |
| `MR00-E03` | MVP gate log | `.venv/bin/python scripts/check_mvp.py --root . --runs runs/mvp` | 50/75, `blocked_with_evidence` |

## Kiểm tra đã chạy

| Test ID | Lệnh | Exit code | Kết quả |
| --- | --- | ---: | --- |
| `MR00-T01` | `git rev-parse HEAD` | 0 | candidate baseline xác định |
| `MR00-T02` | `.venv/bin/python scripts/check_mvp.py --root . --runs runs/mvp` | 1 | expected: artifact v0 bị release contract chặn |

## Việc chưa hoàn tất

- MR-01 phải hoàn thiện lineage implementation, chạy full gate, commit và push.
- MVP v1 scope/evidence/review chưa tồn tại; release bị chặn.

## Sửa đổi phiên trước

Không. Phiên này ghi baseline mới; không sửa artifact hay conclusion của MVP v0.

## Bàn giao

MR-00 hoàn tất. Work package tiếp theo là MR-01: hoàn thiện implementation
lineage, mở rộng negative coverage, chạy full gate, commit và push trước khi
định nghĩa scope release v1.
