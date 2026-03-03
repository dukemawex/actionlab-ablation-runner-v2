from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from actionlab_ablation_runner.orchestrator import run_pipeline, summarize

app = typer.Typer(help="ActionLab ablation runner")


@app.command()
def run(config: Path = Path("config.yaml")) -> None:
    run_pipeline(str(config))
    print("[green]Pipeline completed[/green]")


@app.command("summarize")
def summarize_cmd(config: Path = Path("config.yaml")) -> None:
    message = summarize(str(config))
    print(message)


@app.command("paper")
def paper_cmd(config: Path = Path("config.yaml")) -> None:
    run_pipeline(str(config))
    print("[blue]Paper artifacts generated at artifacts/paper.typ[/blue]")


if __name__ == "__main__":
    app()
