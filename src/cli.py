from io import Writer

import click


@click.group()
def cli():
    pass


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
