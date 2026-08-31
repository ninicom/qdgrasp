"""Config schema and registry entry for QDGrasp-Flow (P4-08).

Three scales are declared -- ``n``, ``s`` and ``m``.  Only ``n`` is required to
train in P4; ``s`` and ``m`` exist so that widening the model later is a config
change with a test behind it rather than an edit to the module that built the
last checkpoint.

The scale table is the *only* place widths and depths are written down.  A
preset names a scale and may override a small set of scalars; anything else is
refused, because ``ModelConfig.params`` is an open mapping and a typo in it
would otherwise be read as "use the default" and silently change what a run
means.

The registered builder returns :class:`QDGraspFlow`, which binds one
:class:`GraspFlowModel` to one robot.  The model itself stays robot-agnostic --
it takes a ``HandGraph`` per call, which is what makes one encoder serve LEAP
and Allegro -- so the binding lives here rather than in the model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import torch
from torch import nn

from qdgrasp.api.results import GraspResults
from qdgrasp.config.registry import register_model
from qdgrasp.config.schema import ConfigError, ModelConfig
from qdgrasp.models.encoder import EncoderConfig
from qdgrasp.models.flow import FlowConfig, GraspFlowModel, GraspPrediction
from qdgrasp.models.hand_graph import HandGraphEncoderConfig
from qdgrasp.models.losses import LossWeights, forward_and_loss, geodesic_rotation_error
from qdgrasp.models.tokenizer import TokenizerConfig
from qdgrasp.robot.graph import HandGraph
from qdgrasp.robot.schema import RobotConfigV2
from qdgrasp.robot.spec import RobotSpec

#: Registry name a ``model`` document must give as its ``type``.
MODEL_TYPE = "qdgrasp_flow"


@dataclasses.dataclass(frozen=True)
class FlowScale:
    """Widths and depths of one named scale."""

    encoder_channels: tuple[int, ...]
    encoder_depths: tuple[int, ...]
    encoder_window: int
    graph_channels: int
    graph_layers: int
    flow_channels: int
    flow_layers: int
    conditioning_layers: int
    heads: int


#: The scale table.  ``n`` is the P4 model; ``s`` and ``m`` are config only.
FLOW_SCALES: dict[str, FlowScale] = {
    "n": FlowScale(
        encoder_channels=(32, 64, 128, 192),
        encoder_depths=(1, 1, 2, 2),
        encoder_window=32,
        graph_channels=128,
        graph_layers=3,
        flow_channels=192,
        flow_layers=3,
        conditioning_layers=2,
        heads=4,
    ),
    "s": FlowScale(
        encoder_channels=(48, 96, 192, 288),
        encoder_depths=(1, 2, 3, 3),
        encoder_window=32,
        graph_channels=192,
        graph_layers=4,
        flow_channels=288,
        flow_layers=4,
        conditioning_layers=3,
        heads=6,
    ),
    "m": FlowScale(
        encoder_channels=(64, 128, 256, 384),
        encoder_depths=(2, 2, 4, 4),
        encoder_window=48,
        graph_channels=256,
        graph_layers=4,
        flow_channels=384,
        flow_layers=6,
        conditioning_layers=3,
        heads=8,
    ),
}

#: Scalars a preset may override on top of its scale.  Kept deliberately small:
#: anything that changes the parameter *shape* belongs in the scale table, where
#: it is named once and shared by every preset that cites it.
OVERRIDABLE_PARAMS: tuple[str, ...] = ("voxel_size", "extent", "flow_steps", "max_joints", "grasps")


@dataclasses.dataclass(frozen=True)
class FlowModelSettings:
    """Everything a preset can say about a QDGrasp-Flow model."""

    scale: str = "n"
    voxel_size: float = 0.005
    extent: float = 0.5
    flow_steps: int = 5
    max_joints: int = 32
    #: How many grasps :meth:`QDGraspFlow.predict_results` draws per call.
    grasps: int = 8

    def validate(self) -> None:
        if self.scale not in FLOW_SCALES:
            known = ", ".join(sorted(FLOW_SCALES))
            raise ConfigError(f"unknown QDGrasp-Flow scale '{self.scale}'; declared scales: {known}")
        if self.grasps <= 0:
            raise ConfigError(f"grasps must be positive, got {self.grasps}")
        # The remaining scalars are validated by the configs they feed, which is
        # where their real constraints live (a grid too fine to pack, a head too
        # narrow for its joints); duplicating those rules here would let the two
        # copies drift.
        self.tokenizer().validate()
        self.encoder().validate()
        self.hand_graph().validate()
        self.flow().validate()

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> FlowModelSettings:
        """Read a preset's ``params`` mapping, refusing anything unrecognised."""

        allowed = {"scale", *OVERRIDABLE_PARAMS}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ConfigError(
                f"unknown QDGrasp-Flow parameters {unknown}; a preset may set {sorted(allowed)}. "
                "Widths and depths come from the scale table, not from params."
            )
        settings = cls(**params)  # type: ignore[arg-type]
        settings.validate()
        return settings

    @property
    def scale_table(self) -> FlowScale:
        return FLOW_SCALES[self.scale]

    def tokenizer(self) -> TokenizerConfig:
        return TokenizerConfig(voxel_size=self.voxel_size, extent=self.extent)

    def encoder(self) -> EncoderConfig:
        table = self.scale_table
        return EncoderConfig(
            channels=table.encoder_channels,
            depths=table.encoder_depths,
            window=table.encoder_window,
            heads=table.heads,
        )

    def hand_graph(self) -> HandGraphEncoderConfig:
        table = self.scale_table
        return HandGraphEncoderConfig(channels=table.graph_channels, layers=table.graph_layers)

    def flow(self) -> FlowConfig:
        table = self.scale_table
        return FlowConfig(
            channels=table.flow_channels,
            heads=table.heads,
            conditioning_layers=table.conditioning_layers,
            flow_layers=table.flow_layers,
            max_joints=self.max_joints,
            flow_steps=self.flow_steps,
        )

    def build(self) -> GraspFlowModel:
        """Instantiate the robot-agnostic model this preset describes."""

        self.validate()
        return GraspFlowModel(
            encoder=self.encoder(), hand=self.hand_graph(), flow=self.flow(), tokenizer=self.tokenizer()
        )


class QDGraspFlow(nn.Module):
    """One QDGrasp-Flow model bound to one robot profile.

    The binding is what the P1 engine expects: it hands a model a batch and a
    point cloud and expects grasps back, with no robot argument in sight.  The
    generative core keeps taking a ``HandGraph`` per call, so binding here costs
    nothing in cross-embodiment terms -- a second robot is a second binding over
    the same weights.
    """

    def __init__(
        self,
        settings: FlowModelSettings,
        robot: RobotSpec,
        *,
        model_hash: str | None = None,
        robot_hash: str | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.robot = robot
        # Results have to name what produced them.  When a document built this
        # model its content hash is authoritative; when a script built it
        # directly there is no document, so the fallback hashes the settings
        # themselves and says so in the prefix rather than emitting an empty
        # string that would read as "unknown provenance".
        self.model_hash = model_hash or self._settings_hash()
        self.robot_hash = robot_hash or f"robot-profile:{robot.config.name}"
        self.model = settings.build()
        joints = len(robot.actuated_joint_names)
        if joints > settings.max_joints:
            raise ConfigError(
                f"robot '{robot.config.name}' has {joints} actuated joints, beyond max_joints="
                f"{settings.max_joints}; widen the head rather than truncating the hand"
            )
        # The graph's tensors are buffers so that ``.to(device)`` carries them
        # with the weights.  They are not persistent: the graph is derived from
        # the robot profile, and a checkpoint that stored a stale copy of it
        # would load a hand that no longer matches its own YAML.
        graph = robot.to_hand_graph()
        self._graph_meta = graph
        self.register_buffer("graph_node_features", graph.node_features, persistent=False)
        self.register_buffer("graph_edge_index", graph.edge_index, persistent=False)
        self.register_buffer("graph_edge_features", graph.edge_features, persistent=False)

    @property
    def graph(self) -> HandGraph:
        """The hand graph, on whichever device the buffers currently live."""

        return HandGraph(
            node_names=self._graph_meta.node_names,
            node_features=self.graph_node_features,
            edge_index=self.graph_edge_index,
            edge_features=self.graph_edge_features,
            palm_index=self._graph_meta.palm_index,
            fingertip_indices=self._graph_meta.fingertip_indices,
            actuated_joint_names=self._graph_meta.actuated_joint_names,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.robot.actuated_joint_names)

    def forward(self, points: torch.Tensor, generator: torch.Generator | None = None) -> GraspPrediction:
        return self.model(points, self.graph, self.robot, generator=generator)

    # -- engine contract ---------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _prediction, losses = self._forward_and_loss(batch)
        return losses.total

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Losses plus the pose errors a loss value cannot be read as."""

        prediction, losses = self._forward_and_loss(batch)
        metrics = {name: value.detach() for name, value in losses.terms.items()}
        metrics["total"] = losses.total.detach()
        metrics["palm_translation_m"] = (
            torch.linalg.norm(prediction.palm_translation - batch["palm_pos"], dim=-1).mean().detach()
        )
        metrics["palm_rotation_rad"] = (
            geodesic_rotation_error(prediction.palm_rotation, batch["palm_rot"]).mean().detach()
        )
        metrics["joint_abs_rad"] = (prediction.joint_angles - batch["joint_angles"]).abs().mean().detach()
        return metrics

    def _forward_and_loss(self, batch: dict[str, torch.Tensor]):
        missing = sorted(
            {"points", "palm_pos", "palm_rot", "joint_angles", "fingertip_positions", "success"} - set(batch)
        )
        if missing:
            raise KeyError(f"batch is missing {missing}")
        return forward_and_loss(
            self.model,
            self.robot,
            self.graph,
            points=batch["points"],
            palm_pos=batch["palm_pos"],
            palm_rot=batch["palm_rot"],
            joint_angles=batch["joint_angles"],
            fingertip_positions=batch["fingertip_positions"],
            success=batch["success"],
            weights=LossWeights(),
        )

    def _settings_hash(self) -> str:
        payload = json.dumps(dataclasses.asdict(self.settings), sort_keys=True, separators=(",", ":"))
        return f"settings:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    @torch.no_grad()
    def predict_results(self, points: torch.Tensor) -> GraspResults:
        """Draw ``settings.grasps`` grasps for one cloud, ranked by quality.

        The flow head is generative, so ``K`` grasps are ``K`` draws from one
        conditioning vector rather than ``K`` heads.  ``seed_points`` reports the
        observed point nearest each palm: this model is not seeded by a point,
        and reporting an invented seed would be worse than reporting the one
        thing about the observation that the grasp is actually near.
        """

        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(f"predict_results expects one cloud [N, 3], got {tuple(points.shape)}")
        batch = points.unsqueeze(0).expand(self.settings.grasps, -1, -1)
        prediction = self.forward(batch)
        score = torch.sigmoid(prediction.quality_logit)
        order = torch.argsort(score, descending=True)
        translation = prediction.palm_translation[order]
        distances = torch.cdist(translation, points)
        seed_points = points[distances.argmin(dim=-1)]
        return GraspResults(
            translation=translation,
            rotation=prediction.palm_rotation[order],
            joint_names=self.joint_names,
            joint_values=prediction.joint_angles[order],
            score=score[order],
            seed_points=seed_points,
            frame=self.robot.config.frame,
            model_hash=self.model_hash,
            robot_hash=self.robot_hash,
        )

    def preprocess_schema(self) -> dict[str, Any]:
        return {
            "input": "points",
            "layout": "[B, N, 3]",
            "dtype": "float32",
            "units": "meters",
            "frame": "object",
            "normalization": "none",
            "voxel_size_m": self.settings.voxel_size,
            "extent_m": self.settings.extent,
        }

    def example_inputs(self) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(0)
        return (torch.randn(1, 256, 3, generator=generator) * 0.05,)


@register_model(MODEL_TYPE)
def build_qdgrasp_flow(model_config: ModelConfig, robot_config: Any) -> QDGraspFlow:
    """Registry entry point for ``type: qdgrasp_flow``."""

    if not isinstance(robot_config, RobotConfigV2):
        raise ConfigError(
            f"QDGrasp-Flow needs a robot/v2 profile with kinematics, got "
            f"{type(robot_config).__name__}; the FK consistency term has nothing to compute without one"
        )
    settings = FlowModelSettings.from_params(dict(model_config.params))
    robot = RobotSpec.from_config(robot_config, sample_anchors=False)
    return QDGraspFlow(
        settings,
        robot,
        model_hash=model_config.content_hash(),
        robot_hash=robot_config.content_hash(),
    )
