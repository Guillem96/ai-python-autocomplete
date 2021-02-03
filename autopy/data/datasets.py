import pickle
import random

import torch
import torch.utils.data as data

from tokenizers import Tokenizer


class CodeCompletionDataset(data.Dataset):

    def __init__(self, examples_file, tokenizer):
        self.tokenizer = tokenizer
        self.examples = pickle.load(open(examples_file, 'rb'))

    def __getitem__(self, idx):
        example = self.examples[idx]
        example = torch.as_tensor(example)

        middle = example.size(0) // 2

        return example[:middle], example[middle:]

    def __len__(self):
        return len(self.examples)
