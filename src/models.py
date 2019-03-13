import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocabulary_size,
                 max_target_length, sos):
        super(Seq2SeqModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.max_target_length = max_target_length
        self.sos = sos

        # self.teacherf_ratio = 0.5
        self.embedding = nn.Embedding(vocabulary_size + 1, embedding_dim)

        self.encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim
        )
        self.decoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
        )
        self.out = nn.Linear(self.hidden_dim, vocabulary_size)

    def forward(self, x, x_lengths, device=None):
        batch_size = len(x_lengths)

        x_ = self.embedding(x)
        x_ = pack_padded_sequence(x_, x_lengths)

        encoder_outputs, hidden = self.encoder(x_)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)

        decoder_input = torch.tensor(
            [[self.sos for _ in range(len(x_lengths))]],
            dtype=torch.long
        )
        if device:
            decoder_input.to(device=device)

        # Collect these for returning and calculating loss
        decoder_outputs = []
        for t in range(self.max_target_length):
            decoder_input = self.embedding(decoder_input)
            decoder_output, hidden = self.decoder(
                decoder_input, hidden)

            decoder_output = decoder_output.squeeze(0)
            # Predict over vocabulary
            output = self.out(decoder_output)
            output = F.log_softmax(output, dim=1)

            # Add to collection for calculating loss
            decoder_outputs.append(output)
            # No teacher forcing: next input is decoder's own current output
            _, topi = output.topk(1)
            decoder_input = torch.tensor(
                [[topi[i][0] for i in range(batch_size)]], dtype=torch.long
            )

        return decoder_outputs
