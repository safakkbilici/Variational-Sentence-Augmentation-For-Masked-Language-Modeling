import torch
import argparse
import numpy as np

from models.bert_mlm import bert_model, bert_dataset, train_bert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mpe', '--max_position_embeddings', type=int, default=514)
    parser.add_argument('-nah', '--num_attention_heads', type=int, default=12)
    parser.add_argument('-nhl', '--num_hidden_layers', type=int, default=6)
    parser.add_argument('-vs', '--vocab_size', type=int, default=52000)
    parser.add_argument('-tvs', '--type_vocab_size', type=int, default=1)
    parser.add_argument('-fc', '--from_checkpoints', default=None)
    parser.add_argument('-c', '--cuda', type=bool, default=True)

    parser.add_argument('-t', '--tokenizer', type=str, default='models/tokenizer')
    parser.add_argument('-d', '--data', type=str, default='data/expanded_corpus.txt')
    parser.add_argument('-bsiz', '--tokenizer', type=int, default=64)
    parser.add_argument('-mlm','--mlm_prob', type=float, default=0.15)

    parser.add_argument('e', '--epochs', type=int)
    parser.add_argument('-co', '--checkpoint_output', type=str, default='models/bert_pretraining')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-ls', '--log_step', type=int, default=1000)

    args = parser.parse_args()

    model = bert_model(args.vocab_size, args.max_position_embeddings, args.num_attention_heads,
                       args.num_hidden_layers, args.type_vocab_size, args.from_checkpoints, args.cuda)

    dataset, data_collator = bert_dataset(args.tokenizer, args.file_path, args.block_size, args.mlm_prob)

    train_bert(model, args.epochs,data_collator, dataset,args.checkpoint_output, args.batch_size, args.log_step)
