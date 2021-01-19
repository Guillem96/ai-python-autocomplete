import click
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFD, StripAccents


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
                                                 Lowercase(), 
                                                 StripAccents()])
    tokenizer.pre_tokenizer = CharDelimiterSplit(' ')
    tokenizer.post_processor = TemplateProcessing(
            single="<cls> $A <sep>",
            pair="<cls> $A <sep> $B:1 <sep>:1",
            special_tokens=[("<cls>", 1), ("<sep>", 2)])

    files = list(map(str, input_path.iterdir()))

    trainer = WordPieceTrainer(
            vocab_size=20522, 
            special_tokens=["<unk>", "<cls>", "<sep>", "<pad>"])

    tokenizer.train(files, trainer)
    tokenizer.save(str(output_path))


if __name__ == '__main__':
    train_tokenizer()

