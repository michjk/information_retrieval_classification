import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from model_module.lstm_classifier import LSTMClassifier

from data_module.data_preprocessor import *

import os
import random

import datetime

from model_module.model_runner import ModelRunner

from utils import load_parameter_from_json, filter_dotdict_class_propoperty, FactoryClass

from data_module.data_writer import LearningLogger

import argparse

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path parameter json file")
param_json_path = parser.parse_args().path

param = load_parameter_from_json(param_json_path)
lstm_parameter = filter_dotdict_class_propoperty(param, LSTMClassifier)

train_data, dev_data, vocab_size, label_size, label_map, pretrained_embedding_weight = load_dataset(
    param.train_dataset_path, param.dev_dataset_path, param.max_text_length, param.embedding_dim, 
    pretrained_word_embedding_name = param.pretrained_word_embedding_name, pretrained_word_embedding_path = param.pretrained_word_embedding_path
)

lstm_parameter.vocab_size = vocab_size
lstm_parameter.label_size = label_size
lstm_parameter.pretrained_embedding_weight = pretrained_embedding_weight

model_factory = FactoryClass(LSTMClassifier, lstm_parameter)

loss_factory = FactoryClass(nn.NLLLoss)

optimizer_param_dict = filter_dotdict_class_propoperty(param, optim.Adam)
optimizer_factory = FactoryClass(optim.Adam, optimizer_param_dict)

#optimizer = optim.Adagrad(update_parameter, lr=1e-3)
#optimizer = optim.RMSprop(update_parameter, lr=parameter.learning_rate, alpha=0.99, eps=1e-8, weight_decay=5e-4)

learning_logger = LearningLogger(label_map, param.result_folder_path, param.saved_model_file_path, param.train_log_folder_path, param.dev_log_folder_path, param.confusion_matrix_folder_path)
model_runner = ModelRunner(model_factory, loss_factory, optimizer_factory, param.epoch, param.batch_size, learning_logger, param.use_gpu)
start_time = time.time()
model_runner.learn(train_data, dev_data)
print("Overall time elapsed {} sec".format(time.time() - start_time))
