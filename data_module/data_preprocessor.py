import numpy as np
import re
import itertools
from collections import Counter
import json

from torchtext import data
import codecs

from sklearn.model_selection import KFold

from utils import dotdict

import torchwordemb

import dill as pickle

import spacy

spacy_nlp = spacy.load('en')

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`@-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class QuestionWrapper(data.Dataset):
    def __init__(self, text_field, question, **kwargs):
        fields = [('text', text_field)]
        question = clean_str(question)
        examples = []
        examples.append(data.Example.fromlist([question], fields))

        super().__init__(examples, fields, **kwargs)

def preprocess_question(question, text_field, transpose = False, use_gpu = False):
    device = -1
    if use_gpu:
        device = None
    question_data = QuestionWrapper(text_field, question)
    _, question_iter = data.Iterator.splits(
        (question_data, question_data), batch_size=len(question_data),
        repeat=False, device = device
    )

    for batch in question_iter:
        text = batch.text
        if transpose:
            text.data.t_()

        return text

def get_label(label_tensor, label_field):
    pred_index = label_tensor.data.max(1)[1]
    pred_index = pred_index.cpu().numpy()[0]
    label_string = label_field.vocab.itos[pred_index+1]

    return label_string

def tokenizer(comment):
    comment = comment.strip()
    comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    comment = clean_str(comment)
    tokenizer_re = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
    comment = ' '.join(tokenizer_re.findall(comment))
    return [x.text for x in spacy_nlp.tokenizer(comment) if x.text != " "]

def tokenizer_2(comment):
    comment = comment.strip()
    comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    comment = clean_str(comment)
    return [x.text for x in spacy_nlp.tokenizer(comment) if x.text != " "]

def load_dataset(train_path, dev_path, max_text_length, embedding_dim, tokenizer = tokenizer, dev_ratio = 0.1, pretrained_word_embedding_name = "glove.6B.300d", pretrained_word_embedding_path = None,
    saved_text_vocab_path = "text_vocab.pkl", saved_label_vocab_path = "label_vocab.pkl"):
    text_field = data.Field(lower=True, tokenize=tokenizer, fix_length=max_text_length)
    label_field = data.Field(sequential=False)

    print('loading data')
    train_data = data.TabularDataset(path=train_path, format='csv', skip_header=True, fields=[("text", text_field), ('label', label_field)])
    dev_data = data.TabularDataset(path=dev_path, format='csv', skip_header=True, fields=[("text", text_field), ('label', label_field)])
    
    print('building vocab')
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)

    vectors = None

    if pretrained_word_embedding_name == "word2vec":
        vocab, vec = torchwordemb.load_word2vec_bin(pretrained_word_embedding_path)
        text_field.vocab.set_vectors(vocab, vec, embedding_dim)
        vectors = text_field.vocab.vectors
    elif "glove" in pretrained_word_embedding_name:
        text_field.vocab.load_vectors(pretrained_word_embedding_name)
        vectors = text_field.vocab.vectors
    
    pickle.dump(text_field, open(saved_text_vocab_path, 'wb'))
    pickle.dump(label_field, open(saved_label_vocab_path, 'wb'))

    vocab_size = len(text_field.vocab)
    print("vocab size ", vocab_size)
    #from zero
    label_size = len(label_field.vocab) - 1

    return train_data, dev_data, vocab_size, label_size, label_field.vocab.itos, vectors




    
