import torch
from torch import nn
from vocab import Vocabulary
from embedding import Embedding

class Decoder(nn.Module):
    def __init__(self, encoding_dim, embedding_dim, hidden_dim, vocab:Vocabulary):
        super(Decoder, self).__init__()
        # params
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.D = 1 # 2 if bidirectional is True

        # Embedding for words
        # self.embeddings = nn.Embedding(len(vocab), embedding_dim, padding_idx = vocab.word_to_idx[vocab.pad])
        # Linear layer to squash images
        self.out = nn.Linear(encoding_dim, hidden_dim)
        # Linear layer
        self.hidden2out = nn.Linear(self.D * hidden_dim, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True)
        # Log softmax
        self.log_softmax = nn.LogSoftmax(dim = 2)

    # images = (batch, 4096)
    def forward(self, images_encoding, lengths):
        # batch, embedding_dim
        squashed_images = self.out(images_encoding)
        batch_size, _ = squashed_images.size()

        # initialise hidden state and cell state
        hidden_state = squashed_images.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_state = squashed_images.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        # cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        sos_input = Embedding.get(self.get_input(batch_size))

        outputs_sequence = list()
        output = sos_input
        for i in range(max(lengths)):
            out, hidden = self.lstm(output, hidden)
            output = self.hidden2out(out)
            outputs_sequence.append(output)
        
        output_tensor = torch.stack(list(outputs_sequence), dim=0)
        # %%
        log_output = self.log_softmax(output_tensor.squeeze())
        # %%
# %%
        return log_output

    def get_input(self, batch_size):
        sos_id = self.vocab.word_to_idx[self.vocab.sos]
        input = torch.tensor(sos_id).unsqueeze(0).repeat(batch_size, 1)
        return input
