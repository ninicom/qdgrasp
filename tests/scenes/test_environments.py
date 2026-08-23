import pytest
from qdgrasp.scenes.environments import get_environment

def test_table_environment():
    supports = get_environment("table")
    assert len(supports) == 1
    assert supports[0].support_id == "table_surface"

def test_bin_environment():
    supports = get_environment("bin")
    assert len(supports) == 5
    assert supports[0].support_id == "bin_bottom"
    assert "bin_wall_fb_0" in [s.support_id for s in supports]

def test_shelf_environment():
    supports = get_environment("shelf")
    assert len(supports) == 3

def test_invalid_environment():
    with pytest.raises(ValueError):
        get_environment("invalid_env")
