import click

import torch

from tokenizers import Tokenizer

import autopy


@click.command()
@click.option('--checkpoint', type=click.Path(dir_okay=False, exists=True),
              required=True)
@click.option('--tokenizer', type=click.Path(dir_okay=False, exists=True),
              required=True)
@click.option('--prompt', prompt='Introduce code to autocomplete')
def evaluate(checkpoint, tokenizer, prompt):

    device = torch.device('cpu')

    # Load trained tokenizer
    tokenizer = Tokenizer.from_file(tokenizer)

    # Load the trained model
    model = autopy.models.LSTMBased(tokenizer)
    model.eval()
    model.to(device)

    checkpoint = torch.load('models/checkpoint-0.pt', map_location=device)
    del checkpoint['optimizer']
    model.load_state_dict(checkpoint['model'])

    input_prompt = tokenizer.encode(prompt).ids
    input_prompt = torch.as_tensor(input_prompt).unsqueeze(0)

    with torch.no_grad():
        prediction = model.generate(input_prompt.to(device))

    print(autopy.decode_sequence(tokenizer, prediction))


if __name__ == '__main__': 
    evaluate()
