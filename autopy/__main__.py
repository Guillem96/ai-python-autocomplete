import click

from autopy.data.download import download
from autopy.data.preprocess import preprocess
from autopy.data.to_examples import to_examples
from autopy.data.train_tokenizer import train_tokenizer


@click.group()
def cmds():
    pass


cmds.add_command(download, name='download')
cmds.add_command(preprocess, name='preprocess')
cmds.add_command(train_tokenizer, name='fit-tokenizer')
cmds.add_command(to_examples, name='generate-examples')


if __name__ == '__main__':
    cmds()

