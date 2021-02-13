import click
import pickle
import random
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
@click.option('--train-output-path', required=True,
              type=click.Path(dir_okay=False))
@click.option('--test-output-path', required=True,
              type=click.Path(dir_okay=False))
@click.option('--test-prob', default=.1, type=float)
@click.option('-t', '--tokenizer', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--sequence-len', default=1024, type=int)
@click.option('--stride', default=10, type=int)
def to_examples(input_path, train_output_path, test_output_path, test_prob,
                tokenizer, sequence_len, stride):
    input_path = Path(input_path)

    train_output_path = Path(train_output_path)
    train_output_path.parent.mkdir(exist_ok=True)

    test_output_path = Path(test_output_path)
    test_output_path.parent.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer)

    train_examples = []
    test_examples = []

    files = list(input_path.iterdir())

    for fname in tqdm.tqdm(files):
        examples = _file_to_examples(fname, tokenizer, sequence_len, stride)
        dst_set = (test_examples 
                   if random.random() < test_prob 
                   else train_examples)
        dst_set.extend(examples)

    print('Training examples:', len(train_examples))
    print('Test examples:', len(test_examples))

    pickle.dump(train_examples, train_output_path.open('wb'))
    pickle.dump(test_examples, test_output_path.open('wb'))


if __name__ == '__main__':
    to_examples()
