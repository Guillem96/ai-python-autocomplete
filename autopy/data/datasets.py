import pickle
import random

import torch
import torch.utils.data as data

from tokenizers import Tokenizer


class CodeCompletionDataset(data.Dataset):

    def __init__(self, examples_file, tokenizer):
        self.tokenizer = tokenizer

        self.sep_id = torch.as_tensor([self.tokenizer.token_to_id('<sep>')])
        self.cls_id = torch.as_tensor([self.tokenizer.token_to_id('<cls>')])

        self.examples = pickle.load(open(examples_file, 'rb'))

    def __getitem__(self, idx):
        example = self.examples[idx]
        example = torch.as_tensor(example)

        middle = int(example.size(0) * .75)

        context = example[:middle]
        context = torch.cat([context, self.sep_id], 0)

        target = example[middle:]
        target = torch.cat([self.cls_id, target, self.sep_id], 0)

        return context, target

    def __len__(self):
        return len(self.examples)


def decode_sequence(tokenizer, ids):
    result = ''
    decoded = tokenizer.decode(ids)

    for w in decoded.split(' '):
        if w.startswith('##'):
            result += w.replace('##', '')
        else:
            result += ' ' + w

    return result

if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('models/tokenizer.json')
    ds = CodeCompletionDataset('data/train-examples.pkl', tokenizer)
    idx = random.randint(0, len(ds))

    ctx, target = ds[idx]

    print(decode_sequence(tokenizer, target.tolist()))

