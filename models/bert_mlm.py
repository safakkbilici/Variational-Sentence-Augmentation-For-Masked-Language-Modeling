import pickle
import os
import torch

from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, BertConfig
from transformers import Trainer, TrainingArguments


def bert_model(vocab_size=52000, max_position_embeddings=514, num_attention_heads=12,
               num_hidden_layers=6, type_vocab_size=1, from_checkpoints=None, cuda=True):
    
    if load_checkpoints != None:
        model = BertForMaskedLM.from_pretrained(from_checkpoints)
    else:
        config = BertConfig(
            vocab_size = vocab_size,
            num_attention_heads = num_attention_heads,
            num_hidden_layers = num_attention_heads,
            type_vocab_size = type_vocab_size,
        )

    if cuda:
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            print("Cannot detect cuda device. Switching cpu")

    return model
        
        
    
