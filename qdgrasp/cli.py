"""Minimal install-time CLI; lifecycle commands are added in Phase 1."""

from __future__ import annotations

import json

import typer

from .runtime import environment_info, require_cuda

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def root() -> None:
    """QDGrasp command-line interface."""


@app.command("env")
def env_command(require_cuda_device: bool = typer.Option(False, "--require-cuda")) -> None:
    """Print the runtime fingerprint and optionally enforce CUDA hardware."""

    if require_cuda_device:
        require_cuda()
    typer.echo(json.dumps(environment_info().to_dict(), indent=2, sort_keys=True))


def main() -> None:
    """Run the QDGrasp command-line application."""

    app()


if __name__ == "__main__":
    main()
