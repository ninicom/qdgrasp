"""Safe filesystem primitives for dataset artifacts.

Manifest paths are data supplied by the artifact, not trusted application
paths.  Resolving them in one place keeps every reader on the same containment
rule and avoids the subtly incorrect string-prefix check (``/data-evil`` starts
with ``/data`` but is not below it).
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..config.schema import ConfigError


def validate_relative_artifact_path(value: str) -> str:
    """Return a canonical POSIX relative path or raise :class:`ConfigError`.

    Both POSIX and Windows absolute/drive syntax are rejected so a manifest has
    the same meaning on every supported host.  Backslashes are intentionally not
    accepted as alternate separators; artifact manifests use canonical ``/``.
    """

    if not isinstance(value, str) or not value:
        raise ConfigError("artifact path must be a non-empty string")
    if "\x00" in value:
        raise ConfigError("artifact path contains a NUL byte")
    if "\\" in value:
        raise ConfigError(f"artifact path must use '/' separators: {value!r}")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ConfigError(f"artifact path must be relative: {value!r}")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ConfigError(f"artifact path may not contain traversal: {value!r}")

    canonical = posix.as_posix()
    if canonical != value:
        raise ConfigError(f"artifact path is not canonical: {value!r}; use {canonical!r}")
    return canonical


def resolve_contained_regular_file(root: str | Path, relative: str) -> Path:
    """Resolve an existing regular file below ``root`` without path escapes."""

    canonical = validate_relative_artifact_path(relative)
    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise ConfigError(f"artifact root is not a directory: {resolved_root}")

    unresolved = resolved_root / canonical
    candidate = unresolved.resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        raise ConfigError(f"artifact path {relative!r} escapes root {resolved_root}")
    try:
        candidate = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"artifact file does not exist: {unresolved}") from exc
    if not candidate.is_relative_to(resolved_root):
        # Resolve a second time with strict=True so a symlink introduced in an
        # intermediate component cannot escape between the lexical and real-path
        # checks.
        raise ConfigError(f"artifact path {relative!r} escapes root {resolved_root}")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise ConfigError(f"cannot stat artifact file {candidate}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise ConfigError(f"artifact path is not a regular file: {candidate}")
    return candidate


def _atomic_replace_bytes(target: Path, payload: bytes) -> None:
    """Write ``payload`` beside ``target`` and atomically replace the target."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Atomically replace ``path`` with ``payload``."""

    _atomic_replace_bytes(Path(path), payload)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace UTF-8 text at ``path``."""

    _atomic_replace_bytes(Path(path), text.encode("utf-8"))


__all__ = (
    "atomic_write_bytes",
    "atomic_write_text",
    "resolve_contained_regular_file",
    "validate_relative_artifact_path",
)
