import json
from io import Reader, Writer

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-m",
    "--model",
    type=str,
    default="Qwen/Qwen3.5-0.8B",
    help="Change the language model used.",
)
@click.option("-c", "--config", type=click.Path(dir_okay=False), default=None)
@click.argument("file", type=click.File("r"))
def run(model: str, config: str | None, file: Reader[str]):
    from inference import SyllogismSolver

    lm = SyllogismSolver(model, config=config)
    responses = lm.solve(json.load(file))
    click.echo_via_pager(json.dumps(responses))


@cli.command()
@click.option(
    "-r",
    "--ref",
    type=click.Path(dir_okay=False, readable=True),
    help="Set the path to reference data.",
)
@click.option(
    "-o",
    "--out",
    type=click.File("w"),
    help="Write output to the file.",
    default="-",
)
@click.argument("prediction", nargs=-1)
def evaluate(ref: str, prediction: tuple[str, ...], out: Writer[str]):
    from evaluation import run_full_scoring

    for pred in prediction:
        run_full_scoring(ref, pred, out)


if __name__ == "__main__":
    cli()
