import argparse
import torch
import numpy as np
import pandas as pd

from models.bert_sentiment import (
    bert_sentiment_model,
    train_sentiment_model,
    get_data_loaders,
    save_sentiment_model
)

from models.bert_ner import(
    get_iterators,
    bert_ner_model,
    train_ner_model
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--downstream_task', type=str)
    parser.add_argument('-m', '--bert_model', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-t', '--tokenizer', type=str)
    parser.add_argument('-e', '--epochs', type=int, default=3)
    parser.add_argument('-ev', '--evaluation', type=bool, default=True)
    parser.add_argument('-sdm', '--save_downstream_model', type=bool, default=False)
    parser.add_argument('-sdo', '--save_downstream_optimizer', type=bool, default=False)
    parser.add_argument('-sdp', '--save_downstream_path', type=str)
    parser.add_argument('-l', '--load_downstream_model', type=bool, default=False)
    parser.add_argument('-lop', '--load_downstream_optimizer', type=bool, default=False)
    parser.add_argument('-dc', '--downstream_checkpoints_path', type=str, default=None)
    parser.add_argument('-c', '--cuda', type=bool, default=True)
    args = parser.parse_args()

    if args.downstream_task == "sequence classification":
        df_train = pd.read_csv(f"{args.dataset}/train.csv")
        df_test = pd.read_csv(f"{args.dataset}/test.csv")

        train_loader, val_loader = get_data_loaders(
            df_train = df_train,
            df_test = df_test,
            tokenizer = args.tokenizer,
            batch_size = args.batch_size
        )

        bert_model, optimizer, scheduler, loss_fn = bert_sentiment_model(
            pretrained = args.bert_model,
            train_dataloader = train_loader,
            epochs = args.epochs,
            cuda = args.cuda,
            load_model = args.load_downstream_model,
            load_optimizer= args.load_downstream_optimizer,
            load_path = args.downstream_checkpoints_path
        )

        bert_model, optimizer = train_sentiment_model(
            model = bert_model,
            train_dataloader = train_loader,
            val_dataloader = val_loader,
            optimizer = optimizer,
            loss_fn = loss_fn,
            epochs = args.epochs,
            evaluation = args.evaluation,
            cuda = args.cuda,
            scheduler = scheduler
        )

        if args.save_downstream_model and args.save_downstream_optimizer:
            save_sentiment_model(
                model = bert_model,
                optimizer = optimizer,
                path = args.save_downstream_path
            )
    
        elif args.save_downstream_model and args.save_downstream_optimizer == False:
            save_sentiment_model(
                model = bert_model,
                path = args.save_downstream_path
            )
    
    elif args.downstream_task == "sequence labeling":
        df_train = pd.read_csv(f"{args.dataset}/train.csv")
        df_test = pd.read_csv(f"{args.dataset}/test.csv")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not args.cuda:
            device = torch.device("cpu")
        
        train_iterator, valid_iterator, test_iterator, TEXT, TAGS = get_iterators(
            df_train = df_train,
            df_test = df_test,
            tokenizer = args.tokenizer,
            batch_size = args.batch_size,
            transformers = True,
            device = device
        )

        bert_model = bert_ner_model(
            model_name = args.bert_model
            output_dim = len(TAGS),
            TEXT = TEXT,
            TAGS = TAGS,
            dropout = 0.2,
            device = device,
            cuda = args.cuda
        )
            
        bert_model = train_ner_model(
            model = bert_model,
            train_iterator = train_iterator,
            val_iterator = valid_iterator,
            eval_metrics = ["acc"],
            epochs = args.epochs
        )
