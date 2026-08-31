import numpy as np

from qdgrasp.scenes.contracts import SceneObservation


def select_target_uniform_visible(obs: SceneObservation, rng: np.random.Generator, min_visibility: float = 0.05) -> str:
    """
    Selects a target uniformly from the objects that have a visibility score >= min_visibility.
    """
    candidates = [obj_id for obj_id, vis in obs.visibility_by_object.items() if vis >= min_visibility]
    if not candidates:
        raise ValueError("No objects meet the minimum visibility threshold.")

    return str(rng.choice(candidates))


def select_target_difficulty_weighted(
    obs: SceneObservation, rng: np.random.Generator, min_visibility: float = 0.05
) -> str:
    """
    Selects a target with probability inversely proportional to its visibility.
    Harder (more occluded) objects have a higher chance of being selected.
    """
    candidates = []
    weights = []

    for obj_id, vis in obs.visibility_by_object.items():
        if vis >= min_visibility:
            candidates.append(obj_id)
            weights.append(1.0 / vis)

    if not candidates:
        raise ValueError("No objects meet the minimum visibility threshold.")

    weights = np.array(weights)
    probs = weights / weights.sum()

    return str(rng.choice(candidates, p=probs))


def select_target_declutter_ordered(obs: SceneObservation, min_visibility: float = 0.05) -> str:
    """
    Selects the easiest target (highest visibility) to systematically declutter the scene.
    """
    candidates = [(vis, obj_id) for obj_id, vis in obs.visibility_by_object.items() if vis >= min_visibility]
    if not candidates:
        raise ValueError("No objects meet the minimum visibility threshold.")

    # Sort by visibility descending, pick the first
    candidates.sort(reverse=True)
    return candidates[0][1]


def get_target_selector(policy: str):
    if policy == "uniform_visible":
        return select_target_uniform_visible
    elif policy == "difficulty_weighted":
        return select_target_difficulty_weighted
    elif policy == "declutter_ordered":
        return select_target_declutter_ordered
    else:
        raise ValueError(f"Unknown target selection policy: {policy}")
