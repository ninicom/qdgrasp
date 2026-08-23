import numpy as np
from qdgrasp.scenes.contracts import SupportGeometrySpec

def create_table_environment(size: tuple[float, float, float] = (1.0, 1.0, 0.05), friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)) -> list[SupportGeometrySpec]:
    """Creates a basic flat table support."""
    T = np.eye(4)
    T[2, 3] = -size[2] / 2.0  # Top surface is at z=0
    return [
        SupportGeometrySpec(
            support_id="table_surface",
            geom_type="box",
            params={"size": list(size), "friction": list(friction)},
            T_world_support=T
        )
    ]

def create_bin_environment(
    inner_size: tuple[float, float, float] = (0.4, 0.3, 0.2),
    wall_thickness: float = 0.02,
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
) -> list[SupportGeometrySpec]:
    """Creates a bin with a bottom and 4 walls."""
    supports = []

    # Bottom
    w, l, h = inner_size
    wt = wall_thickness

    T_bottom = np.eye(4)
    T_bottom[2, 3] = -wt / 2.0
    supports.append(SupportGeometrySpec(
        support_id="bin_bottom",
        geom_type="box",
        params={"size": [w + 2*wt, l + 2*wt, wt], "friction": list(friction)},
        T_world_support=T_bottom
    ))

    # Front and Back walls (along x axis, thickness on y)
    for i, sign in enumerate([1, -1]):
        T_wall = np.eye(4)
        T_wall[1, 3] = sign * (l / 2.0 + wt / 2.0)
        T_wall[2, 3] = h / 2.0
        supports.append(SupportGeometrySpec(
            support_id=f"bin_wall_fb_{i}",
            geom_type="box",
            params={"size": [w + 2*wt, wt, h], "friction": list(friction)},
            T_world_support=T_wall
        ))

    # Left and Right walls (along y axis, thickness on x)
    for i, sign in enumerate([1, -1]):
        T_wall = np.eye(4)
        T_wall[0, 3] = sign * (w / 2.0 + wt / 2.0)
        T_wall[2, 3] = h / 2.0
        supports.append(SupportGeometrySpec(
            support_id=f"bin_wall_lr_{i}",
            geom_type="box",
            params={"size": [wt, l, h], "friction": list(friction)},
            T_world_support=T_wall
        ))

    return supports

def create_shelf_environment() -> list[SupportGeometrySpec]:
    """Creates a shelf with multiple levels."""
    supports = []
    levels = 3
    spacing = 0.3
    size = (0.8, 0.4, 0.02)
    friction = (1.0, 0.005, 0.0001)

    for i in range(levels):
        T_level = np.eye(4)
        T_level[2, 3] = i * spacing
        supports.append(SupportGeometrySpec(
            support_id=f"shelf_level_{i}",
            geom_type="box",
            params={"size": list(size), "friction": list(friction)},
            T_world_support=T_level
        ))

    return supports

def get_environment(env_id: str, **kwargs) -> list[SupportGeometrySpec]:
    if env_id == "table":
        return create_table_environment(**kwargs)
    elif env_id == "bin":
        return create_bin_environment(**kwargs)
    elif env_id == "shelf":
        return create_shelf_environment(**kwargs)
    elif env_id == "custom":
        return []
    else:
        raise ValueError(f"Unknown environment: {env_id}")
