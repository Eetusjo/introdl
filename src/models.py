import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LuongAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.attention = torch.nn.Linear(hidden_size, hidden_size)

    def score(self, hidden, encoder_output):
        energy = self.attention(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        attention_energies = self.score(hidden, encoder_outputs)
        attention_energies = attention_energies.t()
        return F.softmax(attention_energies, dim=1).unsqueeze(1)


class AttentionDecoder(nn.Module):
    def __init__(self, embedding, hidden_size, output_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define layers
        self.embedding = embedding
        self.gru = nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=1
        )
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attention = LuongAttention(hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights
        attn_weights = self.attention(rnn_output, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class Seq2SeqModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocabulary_size,
                 attention, max_target_length, sos):
        super(Seq2SeqModel, self).__init__()
        self.attention = attention
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.max_target_length = max_target_length
        self.sos = sos

        # self.teacherf_ratio = 0.5
        self.embedding_enc = nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding_dec = nn.Embedding(vocabulary_size, embedding_dim)

        self.encoder = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim
        )
        if attention:
            self.decoder = AttentionDecoder(
                self.embedding_dec, hidden_dim, vocabulary_size)
        else:
            self.decoder = nn.GRU(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
            )

        self.out = nn.Linear(self.hidden_dim, vocabulary_size)

    def forward(self, x, x_lengths, device):
        batch_size = len(x_lengths)

        x_ = self.embedding_enc(x)
        x_ = pack_padded_sequence(x_, x_lengths)

        encoder_outputs, hidden = self.encoder(x_)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)

        decoder_input = torch.tensor(
            [[self.sos for _ in range(len(x_lengths))]],
            dtype=torch.long
        )

        # Collect these for returning and calculating loss
        decoder_outputs = []
        for t in range(self.max_target_length):
            if device:
                decoder_input = decoder_input.to(device=device)

            # Should wrap vanilla decoder into class for clarity
            # bc this looks ugly but whatever.
            if self.attention:
                decoder_output, hidden = self.decoder(
                    decoder_input, hidden, encoder_outputs)
            else:
                decoder_input = self.embedding_dec(decoder_input)
                decoder_output, hidden = self.decoder(
                    decoder_input, hidden)
                decoder_output = decoder_output.squeeze(0)
                # Predict over vocabulary
                decoder_output = self.out(decoder_output)
                decoder_output = F.log_softmax(decoder_output, dim=1)

            # Add to collection for calculating loss
            decoder_outputs.append(decoder_output)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.tensor(
                [[topi[i][0] for i in range(batch_size)]], dtype=torch.long
            )

        return decoder_outputs
