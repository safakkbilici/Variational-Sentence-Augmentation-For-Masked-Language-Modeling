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
        outputs = self.bert(input_ids = token_ids, attention_mask=attention_masks)
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
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def get_data_loaders(df_train, df_test, tokenizer, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    all_data = np.concatenate([df_train.sentence.values, df_test.sentence.values])
    encoded_data = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in all_data]
    max_len = max([len(sentence) for sentence in encoded_data])

    train_ids, train_masks = preprocess_data(df_train.sentence.values, tokenizer, max_len)
    val_ids, val_masks = preprocess_data(df_test.sentence.values, tokenizer, max_len)

    train_labels = torch.LongTensor(df_train.target.values)
    val_labels = torch.LongTensor(df_test.target.values)


    train_data = TensorDataset(train_ids, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_ids, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def train_sentiment_model(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, epochs=2, evaluation=True, cuda=True):
    if cuda == True and torch.cuda.is_available() == False:
        print("Cannot detect cuda device. Switching cpu")
    elif cuda == True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("Training started...")

    for epoch in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)

        if evaluation:
            val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)

            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    print("Training complete!")
    return model, optimizer

def evaluate(model, val_dataloader, loss_fn, device):

    model.eval()
    val_accuracy = []
    val_loss = []
    f1_score = []
    precision_score = []
    recall_score = []

    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy


def bert_sentiment_model(pretrained, train_dataloader,epochs=2, cuda=True, load_model=False,
                         load_optimizer=False, load_path="models/bert_sentiment/sentiment.pt"):
    
    if load_model or load_optimizer:
        checkpoint = torch.load(load_path, map_location="cpu")
    if load_model:
        bert_classifier = BERTSentiment(pretrained,freeze_bert=True)
        bert_classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        bert_classifier = BERTSentiment(pretrained,freeze_bert=True)

    
    if cuda == True and torch.cuda.is_available() == False:
        print("Cannot detect cuda device. Switching cpu")
        device = torch.device("cpu")
    elif cuda == True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    bert_classifier.to(device)
        
    optimizer = AdamW(
        bert_classifier.parameters(),
        lr=5e-5,
        eps=1e-8
    )

    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_fn = nn.CrossEntropyLoss()

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return bert_classifier, optimizer, scheduler, loss_fn

def save_sentiment_model(model, optimizer=None, path="models/bert_sentiment/sentiment.pt"):
    if optimizer != None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, path)


    
