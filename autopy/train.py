import click
import functools

import torch

from tokenizers import Tokenizer

import autopy


def _collate_fn(batch, pad_id):
    x = [o[0] for o in batch]
    y = [o[1] for o in batch]

    x = torch.nn.utils.rnn.pad_sequence(x, True, pad_id)
    y = torch.nn.utils.rnn.pad_sequence(x, True, pad_id)

    return x, y


@click.command()
@click.option('--dataset', type=click.Path(exists=True, dir_okay=False))
@click.option('--tokenizer', type=click.Path(exists=True, dir_okay=False))

@click.option('--batch-size', type=int, default=64)
@click.option('--epochs', type=int, default=20)
@click.option('--learning-rate', type=float, default=1e-3)
def train(dataset, tokenizer,
          batch_size, epochs, learning_rate):

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Torch is using device:', device)

    tokenizer = Tokenizer.from_file(tokenizer)

    ds = autopy.data.CodeCompletionDataset(dataset, tokenizer)
    train_ds_len = int(len(ds) * .8)
    idx = torch.randperm(len(dataset))

    train_ds = torch.utils.data.Subset(ds, idx[:train_ds_len])
    valid_ds = torch.utils.data.Subset(ds, idx[train_ds_len:])

    collate_fn = functools.partial(_collate_fn, 
                                   pad_id=tokenizer.token_to_id('<pad>'))
    train_dl = torch.utils.data.DataLoader(train_ds, 
                                           batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)

    valid_dl = torch.utils.data.DataLoader(valid_ds, 
                                           batch_size=batch_size,
                                           shuffle=False, collate_fn=collate_fn)

    model = autopy.models.LSTMBased(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  learning_rate, 
                                  weight_decay=1e-3)

    criterion_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id('<pad>'))

    for epoch in range(epochs):
        autopy.engine.train_one_epoch(model=model, 
                                      optimizer=optimizer, 
                                      data_loader=train_dl, 
                                      criterion_fn=criterion_fn,
                                      epoch=epoch,
                                      device=device)


if __name__ == '__main__':
    train()
