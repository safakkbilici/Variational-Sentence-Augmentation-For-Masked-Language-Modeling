# Variational Sentence Augmentation For Masked Language Modeling

Code for our paper "Variational Sentence Augmentation For Masked Language Modeling" (Innovations in Intelligent Systems and Applications Conference, ASYU 2021). 

**From abstract:** We introduce a variational sentence augmentation method that consists of Variational Autoencoder and Gated Recurrent Unit. The proposed method for data augmentation benefits from its latent space representation, which encodes semantic and syntactic properties of the language. After learning the representation of the language, the model generates sentences from its latent space by sequential structure of GRU. By augmenting existing unstructured corpus, the model improves Masked Language Modeling on pre-training. As a result, it improves fine-tuning as well. In pre-training, our method increases the prediction rate of masked tokens. In fine-tuning, we show that variational sentence augmentation can help semantic tasks and syntactic tasks. We make our experiments and evaluations on a limited dataset containing Turkish sentences, which also stands for a contribution to low resource languages.

## Train On Your Corpus
Organize your folder structure as:
      data---
            |
            -- corpus.train.txt
            |
            -- corpus.valid.txt
        
      
```python3
