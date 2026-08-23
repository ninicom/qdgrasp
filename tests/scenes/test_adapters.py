import pytest
from qdgrasp.scenes.adapters import get_adapter

def test_registry_allowlist():
    # Valid adapters
    native = get_adapter("native")
    g1b = get_adapter("graspnet1b")
    dgn2 = get_adapter("dexgraspnet2")
    gc6d = get_adapter("graspclutter6d")

    assert native is not None
    assert g1b is not None
    assert dgn2 is not None
    assert gc6d is not None

def test_registry_invalid_adapter():
    with pytest.raises(ValueError, match="Unknown scene adapter"):
        get_adapter("invalid_adapter_name")

def test_probe_invalid_root_fails_gracefully(tmp_path):
    adapter = get_adapter("graspnet1b")
    info = adapter.probe(str(tmp_path))
    assert not info.is_valid
    assert info.num_scenes == 0
    assert info.dataset_id == "graspnet-1billion"

def test_index_limits(tmp_path):
    adapter = get_adapter("graspnet1b")
    # GraspNet train has 100 scenes normally, but here we don't have files
    # The adapter statically returns the 100 scenes for split="train"
    index = adapter.index(str(tmp_path), split="train", limit=5)
    assert len(index.scene_keys) == 5
    assert index.scene_keys[0] == "scene_0000"
