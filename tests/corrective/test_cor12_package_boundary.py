"""COR-12: the package ships more surface than it uses or defends.

Three unrelated symptoms of one habit -- shipping a declaration without the
behaviour behind it.

*Padding without a mask.*  A short point cloud is padded with zeros and handed
on with no way to tell padding from geometry, so the model sees points piled at
the origin and learns them.

*Configuration keys that do nothing.*  ``num_workers``, ``pin_memory``,
``drop_last``, ``seed`` and ``robot_profiles`` are accepted by the v2 data
schema and read by nobody, so a config file can describe a run that was never
performed.

*A legacy namespace inside the wheel.*  ``exec``, unrestricted ``eval``,
``weights_only=False`` and ``allow_pickle=True`` are installed by default as
part of the base package, so an optional dependency's history is everybody's
attack surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _corrective_support import characterization

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "qdgrasp"


@characterization(
    "COR-12",
    note="zero padding is indistinguishable from geometry",
    satisfied_by="R2",
)
def test_a_padded_point_cloud_carries_a_mask(verified_corpus) -> None:
    from qdgrasp.dataset.loader import DgnOpenDataset

    dataset = DgnOpenDataset(
        dataset_root=verified_corpus,
        split="train",
        robot_name="leap_hand",
        point_count=4096,
    )
    item = dataset[0]

    assert "point_mask" in item, (
        "a short cloud was padded with zeros and no mask; the tokenizer cannot tell the padding from a real "
        "cluster of points at the origin"
    )
    padded = int((~item["point_mask"]).sum())
    assert padded > 0, "this fixture is only meaningful when padding actually happened"
    assert bool((item["points"][item["point_mask"]] != 0).any())


def _keys_read_from_the_config(source: Path, function: str) -> set[str]:
    """Which fields the builder actually reads off the configuration object."""

    tree = ast.parse(source.read_text(encoding="utf-8"))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    read: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config":
            read.add(node.attr)
        if not isinstance(node, ast.Call):
            continue
        arguments = [item for item in node.args if isinstance(item, ast.Constant) and isinstance(item.value, str)]
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and arguments:
            read.add(arguments[0].value)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and arguments:
            read.add(arguments[0].value)
    return read


@characterization("COR-12", note="v2 data-config keys are read by nobody", satisfied_by="R3")
def test_every_data_config_key_reaches_the_dataset_it_configures() -> None:
    from qdgrasp.dataset.schema import DataConfigV2

    consumed = _keys_read_from_the_config(PACKAGE_ROOT / "dataset" / "loader.py", "create_dgn_open_dataset")
    declared = {name for name in DataConfigV2.model_fields if name not in {"schema_version", "name"}}
    unused = sorted(declared - consumed)

    assert not unused, (
        f"{unused} are accepted by qdgrasp/data/v2 and never read by the builder that consumes it; a key "
        "must either take effect or be refused"
    )


def _dangerous_calls(path: Path) -> list[str]:
    """``exec``/``eval`` calls and pickle-enabling loads, by AST rather than grep."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return [f"{path}: unparseable"]
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            findings.append(f"{path}:{node.lineno}: {node.func.id}()")
        for keyword in node.keywords:
            if not isinstance(keyword.value, ast.Constant):
                continue
            if keyword.arg == "weights_only" and keyword.value.value is False:
                findings.append(f"{path}:{node.lineno}: weights_only=False")
            if keyword.arg == "allow_pickle" and keyword.value.value is True:
                findings.append(f"{path}:{node.lineno}: allow_pickle=True")
    return findings


@characterization("COR-12", note="the packaged namespace still holds exec/eval/unsafe loads")
def test_the_installed_package_has_no_code_execution_surface() -> None:
    findings: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        findings.extend(_dangerous_calls(path))

    assert not findings, "the wheel installs {} code-execution sites:\n  {}".format(
        len(findings), "\n  ".join(item.replace(str(PACKAGE_ROOT.parent) + "/", "") for item in findings[:20])
    )
