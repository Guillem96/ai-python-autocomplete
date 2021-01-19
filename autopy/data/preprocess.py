import click
import string
from pathlib import Path

import tqdm.auto as tqdm


def _read_file(fpath):
    lines = []
    block_comment = False

    with fpath.open() as f:
        for l in f:
            sl = l.strip()
            if len(sl) > 3 and sl.startswith('"""') and sl.endswith('"""'):
                block_comment = False
            elif sl.startswith('"""') or sl.endswith('"""'):
                block_comment = not block_comment
            elif sl == '""""""':
                block_comment = False
            elif not sl.startswith('#') and not block_comment and sl:
                lines.append(l)

    ftext = ''.join(lines)
    ftext = ''.join(c for c in ftext if c in string.printable)
    return ftext


@click.command()
@click.option('-i', '--input-path', help='Directory to the raw python files',
              required=True, type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-path', help='Directory to store the clean code files',
              required=True, type=click.Path(file_okay=False))
def preprocess(input_path, output_path):
    input_path = Path(input_path)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    code_files = list(input_path.iterdir())
    for fname in tqdm.tqdm(code_files):
        ftext = _read_file(fname)
        if ftext:
            fname = fname.stem + fname.suffix
            fname = output_path / fname
            with fname.open('w') as f:
                f.write(ftext)


if __name__ == '__main__':
    preprocess()

