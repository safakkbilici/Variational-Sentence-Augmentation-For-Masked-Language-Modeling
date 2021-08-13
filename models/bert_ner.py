import torch
from transformers import AutoTokenizer
from xtagger import BERTForTagging
from xtagger import df_to_torchtext_data


def get_iterators(df_train, df_test, tokenizer, batch_size, transformers, device):
    train_iterator, valid_iterator, test_iterator, TEXT, TAGS = df_to_torchtext_data(
        df_train, 
        df_test, 
        device, 
        transformers = True,
        tokenizer = tokenizer,
        batch_size=batch_size
    )
    return train_iterator, valid_iterator, test_iterator, TEXT, TAGS

def bert_ner_model(model_name, output_dim, TEXT, TAGS, dropout, device, cuda):
    model = BERTForTagging(
        model_name = model_name,
        output_dim = output_dim,
        TEXT = TEXT,
        TAGS = TAGS,
        dropout = dropout,
        device = device,
        cuda = cuda
    )
    return model

def train_ner_model(model, train_iterator, val_iterator, eval_metrics = ["acc"], epochs = 3):
    model.fit(
        train_iterator = train_iterator, 
        valid_iterator = val_iterator, 
        eval_metrics = eval_metrics, 
        epochs = epochs
    )
    return model

