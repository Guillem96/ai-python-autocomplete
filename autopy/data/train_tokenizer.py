import click
import tqdm.auto as tqdm
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import CharDelimiterSplit


@click.command()
@click.option('-i', '--input-path', required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-path', required=True,
              type=click.Path(dir_okay=False))
def train_tokenizer(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    tokenizer = Tokenizer(WordPiece(unk_token='<unk>'))
    tokenizer.normalizer = normalizers.Sequence([NFD(), 
                                                 StripAccents()])
    tokenizer.pre_tokenizer = CharDelimiterSplit(' ')
    tokenizer.post_processor = TemplateProcessing(
            single="<cls> $A <sep>",
            pair="<cls> $A <sep> $B:1 <sep>:1",
            special_tokens=[("<cls>", 1), ("<sep>", 2)])

    chars = set()
    files = list(map(str, input_path.iterdir()))

    print('Counting unique chars...')
    for f in tqdm.tqdm(files):
        with open(f) as f:
            chars.update(list(f.read()))

    trainer = WordPieceTrainer(
            vocab_size=len(chars) + 10_000, 
            special_tokens=["<unk>", "<cls>", "<sep>", "<pad>"])

    tokenizer.train(files, trainer)
    tokenizer.save(str(output_path))


if __name__ == '__main__':
    train_tokenizer()

