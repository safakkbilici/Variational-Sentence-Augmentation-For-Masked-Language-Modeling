import torch
from datasets import load_dataset, load_metric
from transformers import BertTokenizer
from transformers import AutoTokenizer
from datasets import ClassLabel, Sequence
from transformers import BertForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification


import pandas as pd
import random

metric = load_metric("seqeval")

def tokenize_and_align_labels(tokenizer, examples, label_all_tokens=True):
    tokenzed_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx == None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_ner_data(dataset, tokenizer, label_all_tokens=True):
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True,
                                    fn_kwargs = {'tokenizer': tokenizer, 'label_all_tokens': label_all_tokens})
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return tokenized_dataset, data_collator

def bert_ner_model(pretrained, n_labels, cuda=True):
    if cuda == True and torch.cuda.is_available() == False:
        print("Cannot detect cuda device. Switching cpu")
        device = torch.device("cpu")
    elif cuda == True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = BertForTokenClassification.from_pretrained(pretrained, n_labels)
    model.to(device)
    return model
    
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_ner_model(model,tokenized_dataset, data_collator, tokenizer, batch_size=64,
                    epochs=3, learning_rate=2e-5, weight_decay=0.01, save_steps=10_000, get_token_level_eval = True):
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    args = TrainingArguments(
        "models/bert_ner",
        evaluation_strategy = "epoch",
        learning_rate= learning_rate
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = epochs,
        weight_decay = weight_decay,
        save_steps = save_steps
    )

    trainer = Trainer(
        model,
        args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["validation"],
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics)

    trainer.train()

    if get_token_level_eval:
        predictions, labels, _ = trainer.predict(tokenized_dataset["validation"])
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        print(results)

    return model
