# Python Autocompleter 🐍

Python code generation using Deep Learning NLP techniques.

Encoder-Decoder model for Python Code generation. 🧑💻

## Data gathering & preprocessing

1. **Download the data**: This repository uses the GitHub GraphQL API to generate the
dataset, hence, before start, you have to create a Github API Token from the Developer Settings.
Then run the following command:

```
$ python -m autopy download -o data/raw -u GITHUB_USER -t GITHUB_TOKEN -q tensorflow
```

With the above command, you will download Python files from repositories containing the
word "tensorflow". Then if you want to get more data you can re-run the command but
changing the `-q` argument.

2. **Clean the data**: In this step, we remove the comments and unnecessary trailing spaces.

```
$ python -m autopy preprocess -i data/raw -o data/preprocessed 
```

3. **Train the tokenizer**: Since our data is not exactly natural text, we cannot use
a pretrained vocabulary and either pretrained embeddings. Therefore we train our tokenizer
from scratch. We train a BERT like tokenizer (WordPiece), but instead of pre tokenizing the text
with white spaces, we pre tokenize it at byte level, thus keeping the white spaces which in python are
essential.

```
$ python -m autopy fit-tokenizer -i data/preprocessed -o models/tokenizer.json
```

4. Finally, we convert our cleaned dataset to training examples:

```
$ python -m autopy generate-examples \
    -i data/preprocessed/ \
    --train-output-path data/train-examples.pkl \
    --test-output-path data/train-examples.pkl \
    -t models/tokenizer.json \ 
    --sequence-len 512 # Examples of 1024 tokens
```
