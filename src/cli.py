from io import Writer
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-n",
    type=int,
    default=500,
    help="Set the number of syllogisms to generate.",
)
def generate(n: int):
    """Generate syllogisms in the specified format."""
    from generator import generate_syllogisms

    generate_syllogisms(n)


@cli.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Add training dataset.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    help="Set the path to output directory.",
)
def fine_tune(dataset: list[Path], output: str):
    "Fine-tune an existing model."
    from training import fine_tune

    fine_tune(dataset, output_dir=output)


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
