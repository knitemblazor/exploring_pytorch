import torch.nn as nn
import torch


class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, output_dim,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))

        # hidden = [batch size, hid dim * num directions]

        return output


model = LSTM_net(vocab_size=5, embedding_dim=128, hidden_dim=128, n_layers=3,
                 bidirectional=True, dropout=0.3, pad_idx=0, output_dim=5)

text = torch.tensor([[1, 3, 3, 4, 3],
                     [1, 4, 3, 4, 2],
                     [1, 2, 3, 3, 3]])
text = torch.transpose(text, 0, 1)
output = model(text, torch.tensor([5, 5, 5]))
print(output)