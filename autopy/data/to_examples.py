import click
from pathlib import Path

import tqdm.auto as tqdm

from tokenizers import Tokenizer


def _file_to_examples(fname, tokenizer, sequence_len=512, stride=10):

    with fname.open() as f:
        ftext = f.read()

    ftokens = tokenizer.encode(ftext).ids

    return [ftokens[i: i + sequence_len] for i in range(0, len(ftokens), stride)]


@click.command()
@click.option('-i', '--input-path', required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('-t', '--tokenizer', required=True,
              type=click.Path(exists=True, dir_okay=False))
def to_examples(input_path, tokenizer):
    input_path = Path(input_path)
    tokenizer = Tokenizer.from_file(tokenizer)
    print(dir(tokenizer))
    files = list(input_path.iterdir())
    for fname in tqdm.tqdm(files):
        examples = _file_to_examples(fname, tokenizer)
        print(tokenizer.decode(examples[0]))
        break


if __name__ == '__main__':
    to_examples()

