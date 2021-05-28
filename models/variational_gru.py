# Most part of this code is copy paste from
# https://github.com/timbmg/Sentence-VAE

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils.model_utils import to_var

class VariationalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_dropout, embedding_dropout,
                 latent_size, sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
        super(VariationalGRU, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor = torch.cuda.FloatTensor

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.max_sequence_length = max_sequence_length

        self.latent_size = latent_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.word_dropout_rate = word_dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.encoder_rnn = nn.GRU(embedding_size,
                                  hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional = bidirectional)

        self.decoder_rnn = nn.GRU(embedding_size,
                                  hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional = bidirectional)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        ## Encoder
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        ##

        ## Bottleneck and Reparameterization
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        ##

        ## Decoder
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        if self.word_dropout_rate > 0:
            prob = torch.rand(input_sequence.size()) #randomly masking as in bert
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)

        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        outputs, _ = self.decoder_rnn(packed_input, hidden)
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def inference(self, n=4, z=None):

        if z==None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()
        equence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:
            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)
            output, hidden = self.decoder_rnn(input_embedding, hidden)
            logits = self.outputs2vocab(output)
            input_sequence = self._sample(logits)

            generations = self._save_sample(generations, input_sequence, sequence_running, t)
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)
        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        running_latest = save_to[running_seqs]
        running_latest[:,t] = sample.data
        save_to[running_seqs] = running_latest
        return save_to




