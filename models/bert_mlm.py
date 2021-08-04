import pickle
import os
import torch

from transformers import BertTokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, BertConfig
from transformers import Trainer, TrainingArguments

def bert_model(vocab_size=52000, max_position_embeddings=514, num_attention_heads=12,
               num_hidden_layers=6, type_vocab_size=1, from_checkpoints=None, cuda=True):
    
    if from_checkpoints != None:
        model = BertForMaskedLM.from_pretrained(from_checkpoints)
    else:
        config = BertConfig(
            vocab_size = vocab_size,
            num_attention_heads = num_attention_heads,
            num_hidden_layers = num_attention_heads,
            type_vocab_size = type_vocab_size,
        )

        model = BertForMaskedLM(config)

    if cuda:
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            print("Cannot detect cuda device. Switching cpu")
    return model

def bert_dataset(tokenizer, file_path, block_size=64, mlm_prob=0.15):
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    dataset = LineByLineTextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm=True,
        mlm_probability = mlm_prob
    )
    print(f"Number of samples: {len(dataset.examples)}")
    return dataset, data_collator


def train_bert(model, epochs, data_collator, dataset, checkpoint_output, batch_size, log_step):

    training_args = TrainingArguments(
        output_dir = checkpoint_output,
        num_train_epochs=epochs,
        per_device_train_batch_size = batch_size,
        logging_steps= log_step,
        report_to="none",
        save_steps = len(dataset.examples) // batch_size
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset
    )

    trainer.train()
    trainer.save_model("./models7")
