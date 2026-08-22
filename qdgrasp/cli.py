"""QDGrasp command-line interface: one subcommand per lifecycle stage.

The grammar is ``qdgrasp <stage> --flag value``.  There is no ``key=value``
positional grammar and no implicit alias layer: an unknown flag is an error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import typer

from .api import QDGrasp
from .config.schema import ConfigError
from .export import SUPPORTED_FORMATS
from .runtime import environment_info, require_cuda

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def root(verbose: bool = typer.Option(True, "--verbose/--quiet", help="Log run progress to stderr.")) -> None:
    """QDGrasp command-line interface."""

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command("env")
def env_command(require_cuda_device: bool = typer.Option(False, "--require-cuda")) -> None:
    """Print the runtime fingerprint and optionally enforce CUDA hardware."""

    if require_cuda_device:
        require_cuda()
    typer.echo(json.dumps(environment_info().to_dict(), indent=2, sort_keys=True))


@app.command("train")
def train_command(
    model: str = typer.Option(..., "--model", help="Model YAML path or packaged preset name."),
    data: str = typer.Option(..., "--data", help="Data YAML path or packaged preset name."),
    robot: str = typer.Option(..., "--robot", help="Robot profile YAML path or packaged preset name."),
    device: str = typer.Option("cpu", "--device", help="'cpu' or 'cuda[:index]'; CUDA never falls back to CPU."),
    max_steps: int = typer.Option(100, "--max-steps"),
    stop_after_steps: int | None = typer.Option(None, "--stop-after-steps", help="Session budget within the schedule."),
    batch_size: int = typer.Option(4, "--batch-size"),
    learning_rate: float = typer.Option(1e-3, "--learning-rate"),
    seed: int = typer.Option(0, "--seed"),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
    ema_decay: float = typer.Option(0.0, "--ema-decay"),
    grad_clip: float = typer.Option(0.0, "--grad-clip"),
    val_interval: int = typer.Option(0, "--val-interval"),
    workers: int = typer.Option(0, "--workers"),
    project_dir: str = typer.Option("runs", "--project-dir", help="Relative output root."),
    run_name: str = typer.Option("train", "--run-name"),
    resume: str | None = typer.Option(None, "--resume", help="Resume artifact or run directory."),
    weights: str | None = typer.Option(None, "--weights", help="Public bundle directory to start from."),
) -> None:
    """Train a model and write bundle, resume state and result bundle."""

    grasper = QDGrasp(model, robot=robot, weights=weights, seed=seed)
    result = grasper.train(
        data,
        device=device,
        max_steps=max_steps,
        stop_after_steps=stop_after_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        amp=amp,
        ema_decay=ema_decay,
        grad_clip=grad_clip,
        val_interval=val_interval,
        workers=workers,
        project_dir=project_dir,
        run_name=run_name,
        resume=resume,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))


@app.command("val")
def val_command(
    model: str = typer.Option(..., "--model"),
    data: str = typer.Option(..., "--data"),
    robot: str = typer.Option(..., "--robot"),
    weights: str | None = typer.Option(None, "--weights", help="Public bundle directory to evaluate."),
    device: str = typer.Option("cpu", "--device"),
    batch_size: int = typer.Option(4, "--batch-size"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Evaluate a model on the validation split and print averaged metrics."""

    grasper = QDGrasp(model, robot=robot, weights=weights, seed=seed)
    metrics = grasper.val(data, device=device, batch_size=batch_size, seed=seed)
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


def _load_points(path: str) -> torch.Tensor:
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if "points" not in archive:
                raise ConfigError(f"{source}: archive has no 'points' array")
            array = archive["points"]
    elif source.suffix == ".npy":
        array = np.load(source, allow_pickle=False)
    else:
        raise ConfigError(f"{source}: expected a .npy or .npz point cloud")
    return torch.as_tensor(np.asarray(array, dtype=np.float32))


@app.command("predict")
def predict_command(
    model: str = typer.Option(..., "--model"),
    robot: str = typer.Option(..., "--robot"),
    points: str = typer.Option(..., "--points", help="Path to a .npy/.npz point cloud of shape [N, 3]."),
    weights: str | None = typer.Option(None, "--weights"),
    device: str = typer.Option("cpu", "--device"),
    out: str | None = typer.Option(None, "--out", help="Write the results archive to this path."),
) -> None:
    """Predict ranked grasps for one point cloud."""

    grasper = QDGrasp(model, robot=robot, weights=weights)
    results = grasper.predict(_load_points(points), device=device)
    if out is not None:
        typer.echo(f"saved {results.save(out)}")
    typer.echo(results.summary())


@app.command("export")
def export_command(
    model: str = typer.Option(..., "--model"),
    robot: str = typer.Option(..., "--robot"),
    fmt: str = typer.Option("torchscript", "--format", help=f"One of {', '.join(SUPPORTED_FORMATS)}."),
    weights: str | None = typer.Option(None, "--weights"),
    out_dir: str = typer.Option("runs/export", "--out-dir"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Export weights to TorchScript or ONNX with a metadata sidecar."""

    grasper = QDGrasp(model, robot=robot, weights=weights)
    result = grasper.export(fmt=fmt, out_dir=out_dir, verify=verify)
    typer.echo(json.dumps(result.metadata, indent=2, sort_keys=True))


def main() -> None:
    """Run the QDGrasp command-line application."""

    app()


if __name__ == "__main__":
    main()
