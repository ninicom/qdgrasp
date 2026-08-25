from qdgrasp.scenes.adapters.dexgraspnet2 import DexGraspNet2Adapter
from qdgrasp.scenes.adapters.graspclutter6d import GraspClutter6DAdapter
from qdgrasp.scenes.adapters.graspnet1b import GraspNet1BillionAdapter
from qdgrasp.scenes.adapters.native import NativeAdapter
from qdgrasp.scenes.contracts import SceneAdapter

ADAPTER_REGISTRY: dict[str, type[SceneAdapter]] = {
    "native": NativeAdapter,
    "graspnet1b": GraspNet1BillionAdapter,
    "dexgraspnet2": DexGraspNet2Adapter,
    "graspclutter6d": GraspClutter6DAdapter,
}


def get_adapter(name: str) -> SceneAdapter:
    if name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown scene adapter: {name}. Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[name]()
