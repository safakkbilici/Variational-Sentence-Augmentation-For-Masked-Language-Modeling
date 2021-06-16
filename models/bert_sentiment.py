import pandas as pd
import numpy as np

import re
import torch
import random
import time
import tokenizers
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

class BERTSentiment(nn.Module):
    def __init__(self, pretrained: str, freeze_bert = True):
        super(BERTSentiment, self).__init__()
        D_in, H, D_out = 768, 50, 3

        self.bert = BertModel.from_pretrained(pretrained)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, token_ids, attention_masks):
        outputs = self.bert(input_ids = token_ids, attention_masks=attention_masks)
        cls_contextual_rep = outputs[0][:, 0, :]
        logits = self.classifier(cls_contextual_rep)
        return logits


def regex_tokenizer(string):
    string = re.sub(r'(@.*?)[\s]', ' ', string)
    string = re.sub(r'&amp;', '&', string)
    string = re.sub(r'\s+', ' ', string).strip()
    return string


def preprocess_data(data, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for sentence in data:
        encoded_sent = tokenizer.encode_plus(
            text=regex_tokenizer(sentence),
            add_special_tokens = True,
            max_length=max_len,
            pad_to_max_length = True,
            return_attention_mask = True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_masks'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def get_data_loaders(df_train, df_test, tokenizer, batch_size=32):
    all_data = np.concatenate([df_train.target.values, df_test.target.values])
    encoded_data = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in all_data]
    max_len = max([len(sentence) for sentence in encoded_tweets])

    train_ids, train_masks = preprocess_data(df_train.target.values, tokenizer, max_len)
    val_ids, val_masks = preprocess_data(df_test.target.values, tokenizer, max_len)

    train_labels = torch.LongTensor(df_train.target.values)
    val_labels = torch.LongTensor(df_test.target.values)


    train_data = TensorDataset(train_ids, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset
            
