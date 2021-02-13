import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer


class Attention(nn.Module):

    def __init__(self, hidden_size, units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, units, bias=False)
        self.W2 = nn.Linear(hidden_size, units, bias=False)
        self.V =  nn.Linear(units, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (BATCH, T, HIDDEN_SIZE)
        # decoder_hidden: (BATCH, HIDDEN_SIZE)

        # Add time axis to decoder hidden state
        # in order to make operations compatible with encoder_out
        # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
        decoder_hidden_time = decoder_hidden.unsqueeze(1)

        # uj: (BATCH, T, ATTENTION_UNITS)
        # Note: we can add the both linear outputs thanks to broadcasting
        uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
        uj = torch.tanh(uj)

        # uj: (BATCH, T, 1)
        uj = self.V(uj)

        # Attention mask over inputs
        # aj: (BATCH, T, 1)
        aj = F.softmax(uj, dim=1)

        # di_prime: (BATCH, HIDDEN_SIZE)
        di_prime = aj * encoder_out
        di_prime = di_prime.sum(1)

        return di_prime, uj.squeeze(-1)


class LSTMBased(nn.Module):

    def __init__(self, 
                 tokenizer, 
                 embedding_dim=256,
                 encoder_hidden_size=256,
                 decoder_hidden_size=512,
                 max_decoder_timesteps=512,
                 decoder_strategy='greedy'):

        super(LSTMBased, self).__init__()
        vocab_size = tokenizer.get_vocab_size()
        pad_idx = tokenizer.token_to_id('<pad>')

        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_decoder_timesteps = max_decoder_timesteps
        self.criterion_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, 
                                                reduction='none')
        self.decoder_strategy = decoder_strategy

        if self.decoder_strategy not in {'greedy'}:
            raise ValueError(f'No valid decoder_strategy '
                             f'{self.decoder_strategy}')

        # Encoder
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim, 
                                       padding_idx=pad_idx)

        self.encoder = nn.LSTM(embedding_dim, 
                               encoder_hidden_size,
                               num_layers=1, batch_first=True,
                               bidirectional=True)

        # Decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_input_size = encoder_hidden_size * 2 + self.embedding_dim

        self.decoder_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                               embedding_dim=embedding_dim, 
                                               padding_idx=pad_idx)

        self.attention = Attention(self.decoder_hidden_size, 10)

        self.decoder = nn.LSTM(self.decoder_input_size,
                               self.decoder_hidden_size,
                               num_layers=1, 
                               batch_first=True)

        # Classifier layer
        self.vocab_linear = nn.Linear(self.decoder_hidden_size, vocab_size)

    def _init_decoder_state(self, x):
        state_size = 1, x.size(0), self.decoder_hidden_size
        decoder_hs = torch.zeros(*state_size, device=x.device)
        decoder_cs = torch.zeros(*state_size, device=x.device)
        return decoder_hs, decoder_cs

    def _build_initial_decoder_in(self, x):
        decoder_in = torch.as_tensor(self.tokenizer.token_to_id('<cls>'),
                                     device=x.device)
        decoder_in = decoder_in.unsqueeze(0).repeat(x.size(0), 1)
        decoder_in = self.decoder_embeddings(decoder_in)
        return decoder_in

    def _build_initial_output(self, x):
        output = torch.as_tensor(self.tokenizer.token_to_id('<cls>'),
                                 device=x.device)
        output = output.unsqueeze(0).repeat(x.size(0), 1)
        return output.int()

    def _encoder_forward(self, x):
        x = self.embeddings(x)
        encoder_hidden_state, _ = self.encoder(x)
        return encoder_hidden_state

    def _decoder_step(self, x, decoder_state, encoder_hidden_state):
        decoder_hs, decoder_cs = decoder_state
        context_vector, _ = self.attention(encoder_hidden_state, decoder_hs[0])
        context_vector = context_vector.unsqueeze(1)

        decoder_in = torch.cat([x, context_vector], dim=-1)

        decoder_out, decoder_state = self.decoder(
            decoder_in, (decoder_hs, decoder_cs))

        prediction = self.vocab_linear(decoder_out.squeeze(1))

        return prediction, decoder_state

    def _forward(self, x, y):
        y = y[:, :self.max_decoder_timesteps]

        non_padding_mask = (y != self.tokenizer.token_to_id('<pad>')).float()
        target_lengths = non_padding_mask.sum(-1)

        encoder_hidden_state = self._encoder_forward(x)

        # Initialize decoder state
        decoder_in = self._build_initial_decoder_in(x)
        decoder_state = self._init_decoder_state(x)
        output = self._build_initial_output(x)

        loss = torch.zeros((x.size(0), 1), device=x.device)

        for t in range(1, min(self.max_decoder_timesteps, y.size(1))):

            prediction, decoder_state = self._decoder_step(
                decoder_in, decoder_state, encoder_hidden_state)

            current_loss = self.criterion_fn(prediction, y[:, t]).unsqueeze(-1)
            loss = torch.cat([loss, current_loss], -1)

            decoder_in = y[:, t].view(-1, 1)
            decoder_in = self.decoder_embeddings(decoder_in)

            output = torch.cat([output, 
                                prediction.argmax(1).view(-1, 1).int()], 1)

        loss = non_padding_mask * loss
        loss = loss.sum(-1) / target_lengths
        return loss.mean(), output

    def _generate(self, x):

        encoder_hidden_state = self._encoder_forward(x)

        # Initialize decoder state
        decoder_in = self._build_initial_decoder_in(x)
        decoder_state = self._init_decoder_state(x)

        output = self._build_initial_output(x)

        for t in range(1, self.max_decoder_timesteps):

            prediction, decoder_state = self._decoder_step(
                decoder_in, decoder_state, encoder_hidden_state)

            decoder_in = prediction.argmax(1).view(-1, 1)
            decoder_in = self.decoder_embeddings(decoder_in)
            output = torch.cat([output, 
                                prediction.argmax(1).view(-1, 1).int()], 1)

        return output

    def forward(self, x, y):
        return self._forward(x, y)

    def generate(self, x)
        return self._generate(x)


if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('models/tokenizer.json')
    x = torch.randint(0, 100, size=(2, 256,))
    y = torch.randint(0, 100, size=(2, 256,))

    model = LSTMBased(tokenizer)
    model.train()
    loss, out = model(x, y)
    print(loss, out.size(), out.dtype)
