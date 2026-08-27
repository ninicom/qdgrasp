"""Can the P3.4 pipeline produce a dynamic positive with a real hand?

Measure before building a generator. Uses the validated generated_reachable
object, which is the one geometry known to yield static positives.
"""
import sys, numpy as np, mujoco
from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget
from qdgrasp.dataset.pipeline.generated_reachable import build_generated_reachable_object
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model
from qdgrasp.dynamic.safety import SceneRoles
from qdgrasp.dynamic.primitives import Primitive, PrimitiveKind, TransitionCondition
from qdgrasp.dynamic.static_seeded import run_static_seeded_rollout, SeedPose, RolloutLimits
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

HAND = sys.argv[1]
CFG = {"leap_hand":"leap_hand.yaml","wonik_allegro":"wonik_allegro.yaml","shadow_hand":"shadow_hand.yaml"}[HAND]

spec = RobotSpec.from_config(CFG, sample_anchors=False)
fixture = build_generated_reachable_object(HAND)
model = build_rollout_scene_model(
    resolve_robot_asset(spec.config.source_asset), fixture.collision_geoms,
    object_pos=fixture.object_pos, object_mass=fixture.mass,
)

def gname(i):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i); return n or f"geom_{i}"
def bname(i):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i); return n or f"body_{i}"

target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
target_geoms, support_geoms, robot_geoms = set(), set(), set()
for g in range(model.ngeom):
    b = int(model.geom_bodyid[g])
    if b == target_body: target_geoms.add(g)
    elif g == floor_geom: support_geoms.add(g)
    else: robot_geoms.add(g)

print(f"{HAND}: nq={model.nq} nu={model.nu} target_geoms={len(target_geoms)} robot_geoms={len(robot_geoms)}")

# Allow all robot self-contact: a multi-finger hand touches itself constantly and
# that is not the safety question this probe is asking.
allow = frozenset((min(a,b),max(a,b)) for a in robot_geoms for b in robot_geoms)
roles = SceneRoles(frozenset(target_geoms), frozenset(support_geoms), frozenset(), frozenset(robot_geoms), allow)
budget = ContactSafetyBudget("probe", HAND, 50.,30.,5.,3.,10.,2.,0.005,100.,20.,50.,0.02,0.3,0.2)

data = mujoco.MjData(model); mujoco.mj_forward(model, data)
# Declared open/closed endpoints from the validated release recipe, so grip
# interpolates a real pregrasp-to-close motion instead of slamming to the stops.
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe
recipe = build_release_grasp_recipe(CFG)
initial = recipe.rollout_kwargs["initial_joint_targets"]
closed = recipe.rollout_kwargs["joint_targets"]
open_ctrl = np.zeros(model.nu); closed_ctrl = np.zeros(model.nu)
named = 0
for a in range(model.nu):
    an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    jn = an
    tid = int(model.actuator_trnid[a][0])
    if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, tid) or an
    if jn in initial:
        open_ctrl[a] = initial[jn]; closed_ctrl[a] = closed[jn]; named += 1
print(f"  mapped {named}/{model.nu} actuators to recipe joint targets")
# Seed the hand at the validated pregrasp pose: palm pose from the recipe,
# backed off along the pregrasp direction, with the free root and the mocap
# body both placed there.
from scipy.spatial.transform import Rotation
kw = recipe.rollout_kwargs
palm_pos = np.asarray(kw["palm_pos"], dtype=float)
palm_rot = np.asarray(kw["palm_rot"], dtype=float)
back = np.asarray(kw["pregrasp_direction"], dtype=float) * float(kw["pregrasp_distance"])
root_pos = palm_pos - back
quat_xyzw = Rotation.from_matrix(palm_rot).as_quat()
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

qpos = np.array(data.qpos)
root_joint = None
for j in range(model.njnt):
    if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
        b = int(model.jnt_bodyid[j])
        if b != target_body:
            root_joint = j; break
if root_joint is not None:
    adr = int(model.jnt_qposadr[root_joint])
    qpos[adr:adr+3] = root_pos
    qpos[adr+3:adr+7] = quat_wxyz
    print(f"  seeded hand root at {root_pos.round(3)}")

seed = SeedPose(qpos, np.zeros(model.nu), f"probe:{HAND}",
                open_ctrl=open_ctrl, closed_ctrl=closed_ctrl,
                mocap_pos=root_pos, mocap_quat=quat_wxyz)

prims = (
    Primitive(kind=PrimitiveKind.CAGE, direction=np.asarray(kw["pregrasp_direction"], dtype=float),
              speed=0.05, max_duration_s=1.6, grip=0.2, until=TransitionCondition.TARGET_CONTACT_MADE),
    Primitive(kind=PrimitiveKind.SQUEEZE, direction=np.array([0.,0.,1.]), speed=0.0,
              max_duration_s=0.8, grip=1.0),
    Primitive(kind=PrimitiveKind.LIFT, direction=np.array([0.,0.,1.]), speed=0.10,
              max_duration_s=1.0, grip=1.0, until=TransitionCondition.SUPPORT_RELEASED),
)
traj, out = run_static_seeded_rollout(
    model, roles=roles, budget=budget, seed=seed, primitives=prims,
    horizon=200, control_dt=0.01, limits=RolloutLimits(),
)
from collections import Counter
print("  classes :", dict(Counter(e.contact_class.value for e in traj.contact_graph)))
print("  outcome :", out.failure_stage, "/", out.failure_reason)
print("  terms   :", {k: round(v,5) for k,v in out.objective_terms.items()})
