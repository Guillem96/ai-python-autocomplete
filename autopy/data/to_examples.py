import click
import pickle
from pathlib import Path

import tqdm.auto as tqdm

from tokenizers import Tokenizer


def _file_to_examples(fname, tokenizer, sequence_len=512, stride=10):

    with fname.open() as f:
        ftext = f.read()

    ftokens = tokenizer.encode(ftext).ids

    return [ftokens[i: i + sequence_len] 
            for i in range(0, len(ftokens), stride)]


@click.command()
@click.option('-i', '--input-path', required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-path', required=True, 
              type=click.Path(dir_okay=False))
@click.option('-t', '--tokenizer', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--sequence-len', default=1024, type=int)
@click.option('--stride', default=10, type=int)
def to_examples(input_path, output_path, tokenizer, sequence_len, stride):
    input_path = Path(input_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer)

    all_examples = []
    files = list(input_path.iterdir())

    for fname in tqdm.tqdm(files):
        examples = _file_to_examples(fname, tokenizer, sequence_len, stride)
        all_examples.extend(examples)

    pickle.dump(all_examples, output_path.open('wb'))


if __name__ == '__main__':
    to_examples()

