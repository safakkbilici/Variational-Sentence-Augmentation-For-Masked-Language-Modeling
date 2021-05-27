# Most part of this code is copy paste from
# https://github.com/timbmg/Sentence-VAE

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class VariationalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_dropout, embedding_dropout,
                 latent_size, sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1):
        super(VariationalGRU, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor = torch.cuda.FloatTensor

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        


