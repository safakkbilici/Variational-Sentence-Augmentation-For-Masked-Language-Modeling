# Variational Sentence Augmentation For Masked Language Modeling

Code for our paper "Variational Sentence Augmentation For Masked Language Modeling" (Innovations in Intelligent Systems and Applications Conference, ASYU 2021). 

**From abstract:** We introduce a variational sentence augmentation method that consists of Variational Autoencoder and Gated Recurrent Unit. The proposed method for data augmentation benefits from its latent space representation, which encodes semantic and syntactic properties of the language. After learning the representation of the language, the model generates sentences from its latent space by sequential structure of GRU. By augmenting existing unstructured corpus, the model improves Masked Language Modeling on pre-training. As a result, it improves fine-tuning as well. In pre-training, our method increases the prediction rate of masked tokens. In fine-tuning, we show that variational sentence augmentation can help semantic tasks and syntactic tasks. We make our experiments and evaluations on a limited dataset containing Turkish sentences, which also stands for a contribution to low resource languages.

## Train On Your Corpus
Organize your folder structure as:
```
      data---
            |
            -- corpus.train.txt
            |
            -- corpus.valid.txt
```

then
      
```bash
python3 train_vae.py --data_name "corpus" --print_every 50 --epochs 1
```

for more detailed arguments, see the [source file](https://github.com/safakkbilici/Variational-Sentence-Augmentation-For-Masked-Language-Modeling/blob/main/train_vae.py).

## Generate New Sentences

```bash
python3 augment.py  --data_name "corpus" \
                    --checkpoint "/models/vae_epoch{epoch}.pt" \
                    --generate_iteration 100 --unk_threshold 0
```
for more detailed arguments, see the [source file](https://github.com/safakkbilici/Variational-Sentence-Augmentation-For-Masked-Language-Modeling/blob/main/augment.py).

The augmented sentences are saved in ```augmentations.txt```. Merge this file with original corpus.


## Increase The Performance Of Pretraining

```bash
python3 pretrain_bert.py --epochs 1 \
                         --tokenizer "./tokenizer" \
                         --data "data/corpus.joined.txt"
```
for more detailed arguments, see the [source file](https://github.com/safakkbilici/Variational-Sentence-Augmentation-For-Masked-Language-Modeling/blob/main/pretrain_bert.py).

## Increase The Performance Of Finetuning (Sequence Classification)

Prepare you dataframe (example):

```python

import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv',names = ['sentence','target'])
df_test = pd.read_csv('test.csv', names = ['sentence','target'])

df_train['target'] = df_train['target'].astype(np.float16)
df_test['target'] = df_test['target'].astype(np.float16)

df_train.to_csv("train.csv",index=False)
df_test.to_csv("test.csv",index=False)
```

Then finetune pretrained BERT

```bash
python3 finetune_bert.py --downstream_task "sequence classification" \
                         --bert_model "./models7" \
                         --dataset "." \
                         --tokenizer "./tokenizer"
```
## Increase The Performance Of Finetuning (Sequence Labeling)

Prepare your dataframe (example):

```python
from datasets import load_dataset
dataset = load_dataset("wikiann", "tr")

ner_encoding = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}


train_tokens = []
train_tags = []
for sample in dataset["train"]:
  train_tokens.append(' '.join(sample["tokens"]))
  train_tags.append(' '.join([ner_encoding[a] for a in sample["ner_tags"]]))

test_tokens = []
test_tags = []
for sample in dataset["test"]:
  test_tokens.append(' '.join(sample["tokens"]))
  test_tags.append(' '.join([ner_encoding[a] for a in sample["ner_tags"]]))

df_train = pd.DataFrame({"sentence": train_tokens, "tags": train_tags})
df_test = pd.DataFrame({"sentence": test_tokens, "tags": test_tags})

df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)
```

Then finetune pretrained BERT

```bash
python3 finetune_bert.py --downstream_task "sequence labeling" 
                         --bert_model "./models7" \
                         --dataset "." \
                         --tokenizer "./tokenizer"
```

### Authors

- M. Şafak Bilici
- Mehmet Fatih Amasyali
