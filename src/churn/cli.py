"""``python -m churn <stage>``: the commands dvc.yaml runs, plus report and serve."""
from __future__ import annotations

import json
import logging

import click

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
@click.option("--params", "params_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def cli(ctx, params_path):
    """Churn model pipeline."""
    ctx.obj = config.load_params(params_path)


def _stage(name):
    @cli.command(name)
    @click.pass_obj
    def cmd(params):
        from . import stages

        getattr(stages, f"stage_{name}")(params)

    cmd.__doc__ = f"DVC stage: {name}"
    return cmd


for _name in ("generate", "prepare", "train", "evaluate", "monitor"):
    _stage(_name)


@cli.command()
@click.option("--out", default=str(config.IMAGES))
def report(out):
    """Charts for the README from metrics/ and plots/."""
    from pathlib import Path

    from .report import render_all

    for p in render_all(Path(out)):
        click.echo(f"wrote {p}")


@cli.command()
def summary():
    """Print the metrics files."""
    for name in ("data", "train", "eval", "drift"):
        p = config.METRICS / f"{name}.json"
        if p.exists():
            click.echo(f"--- metrics/{name}.json\n{json.dumps(json.loads(p.read_text()), indent=2)}")
    p = config.REPORTS / "drift_report.json"
    if p.exists():
        d = json.loads(p.read_text())
        click.echo("--- drift reasons\n" + json.dumps(d["reasons"]) + "\n--- feature PSI\n"
                   + json.dumps({f["feature"]: f["psi"] for f in d["features"]}))


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
def serve(host, port):
    """Run the prediction API."""
    import uvicorn

    uvicorn.run("churn.api:app", host=host, port=port, log_level="info")


def main():
    cli()
