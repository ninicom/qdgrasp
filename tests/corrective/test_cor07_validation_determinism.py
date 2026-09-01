"""COR-07: validation is not a measurement of the run it interrupts.

Three separate ways the loop reports something other than what it did.

*The metric is an average of averages.*  ``Runner.validate`` divides each batch's
metric by the number of batches, so with a dataset that does not divide evenly
the last, short batch is weighted like a full one.  The number then depends on
``batch_size``, which is a runtime knob, not a property of the model.

*Validation is not isolated from training.*  It draws from the same RNG the
training loop is using, so how often you look changes what you get.

*EMA is computed and discarded.*  The shadow weights are updated every step and
then neither validated nor saved, so ``ema_decay`` is a setting that costs time
and changes nothing.
"""

from __future__ import annotations

from typing import Any

import torch
from _corrective_support import characterization
from torch import nn


def _runner(batch_size: int):
    from qdgrasp.api import QDGrasp
    from qdgrasp.config import RunConfig, resolve_runtime
    from qdgrasp.engine.callbacks import CallbackList
    from qdgrasp.engine.runner import Runner

    grasper = QDGrasp()
    run_config = RunConfig(batch_size=batch_size)
    runner = Runner(
        run_config=run_config,
        runtime=resolve_runtime(run_config),
        model_config=grasper.model_config,
        robot_config=grasper.robot_config,
        callbacks=CallbackList([]),
    )
    return grasper, runner


def _samples(robot_config: Any, count: int) -> list[dict[str, torch.Tensor]]:
    from qdgrasp.dummy.data import DummyPointDataset

    dataset = DummyPointDataset(samples=count, num_points=16, seed=7, robot_config=robot_config, split="val")
    return [dataset[index] for index in range(count)]


class _DrawingModel(nn.Module):
    """A model whose validation draws, exactly as the flow head's does."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        noise = torch.randn(batch["points"].shape[0])
        return {"loss": (self.scale * noise).mean()}


@characterization("COR-07", note="the metric averages batch means instead of samples")
def test_the_metric_does_not_depend_on_the_batch_size() -> None:
    grasper, small = _runner(batch_size=2)
    _grasper, whole = _runner(batch_size=5)
    dataset = _samples(grasper.robot_config, 5)

    from_small = small.validate(grasper.module, dataset)
    from_whole = whole.validate(grasper.module, dataset)

    assert abs(from_small["loss"] - from_whole["loss"]) < 1e-6, (
        f"the same model on the same five samples reports {from_small['loss']} at batch_size=2 and "
        f"{from_whole['loss']} at batch_size=5; averaging batch means over-weights the short last batch"
    )


@characterization("COR-07", note="validation consumes the training RNG")
def test_validation_does_not_consume_the_training_rng() -> None:
    grasper, runner = _runner(batch_size=2)
    dataset = _samples(grasper.robot_config, 4)
    model = _DrawingModel()

    torch.manual_seed(1234)
    before = torch.get_rng_state()
    runner.validate(model, dataset)
    after = torch.get_rng_state()

    assert torch.equal(before, after), (
        "validation advanced the global RNG, so the training trajectory depends on how often it was "
        "measured; validation needs its own stream"
    )


@characterization("COR-07", note="validate() forces the model back into train mode")
def test_validation_leaves_the_model_in_the_mode_it_found_it() -> None:
    grasper, runner = _runner(batch_size=2)
    dataset = _samples(grasper.robot_config, 4)

    grasper.module.eval()
    runner.validate(grasper.module, dataset)

    assert not grasper.module.training, (
        "validate() switched an evaluating model into train mode on its way out; a helper may not decide "
        "the caller's mode for it"
    )


@characterization("COR-07", note="EMA is updated but never used")
def test_ema_weights_are_the_weights_that_get_published() -> None:
    from safetensors.torch import load_file as load_safetensors

    from qdgrasp.api import QDGrasp

    grasper = QDGrasp()
    result = grasper.train(
        "dummy-tiny.yaml",
        max_steps=4,
        ema_decay=0.9,
        run_name="corrective-ema",
        project_dir="runs",
    )
    bundle = load_safetensors(f"{result.artifacts['bundle']}/weights.safetensors")
    live = {key: value.detach() for key, value in grasper.module.state_dict().items()}

    published_live = all(torch.allclose(bundle[key], live[key]) for key in live)
    assert not published_live, (
        "ema_decay was on, yet the published bundle holds the live weights; either the bundle publishes the "
        "EMA shadow or the option does nothing and should not exist"
    )
