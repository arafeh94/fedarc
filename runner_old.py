import os
import sys

import numpy as np
import psutil
import setproctitle
import torch

import runner_methods
from fedml_api.fedavg.SQLProvider import SQLDataProvider

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from fedml_api.model.linear.lr import LogisticRegression
from runner_methods import *


class args:
    def __init__(self):
        self.sql_host = "localhost"
        self.sql_user = "root"
        self.sql_password = "root"
        self.sql_database = "mnist"


database_clients = [0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]

print("total number of participating trainers:", len(database_clients))
model_stats = {}
models = {}
sample_dict = {}
data_dict = {}

test_data = SQLDataProvider(args()).cache(99999)
print("test data:", test_data.y)
for index, client_idx in enumerate(database_clients):
    data = SQLDataProvider(args()).cache(client_idx)
    model = LogisticRegression(28 * 28, 10)
    trained = train(model, data.batch(8))
    data_dict[index] = data
    model_stats[index] = trained
    models[index] = model
    sample_dict[index] = len(data)
    print("model accuracy:", infer(model, test_data.batch(8)))


def test_a_case(test_case, title='start evaluation'):
    print('-----------------' + title + '-----------------')
    global_model_stats = aggregate(dict_select(test_case, model_stats), dict_select(test_case, sample_dict))
    global_model = LogisticRegression(28 * 28, 10)
    load(global_model, global_model_stats)
    print("test case:", test_case)
    acc_loss = infer(global_model, test_data.batch(8))
    print("global model accuracy:", acc_loss[0])
    print("global model loss:", acc_loss[1])
    print("----------------------------------------------------")


test_cases = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 10, 11, 12, 13, 15, 16],
    [1, 2, 13, 15, 16, 17, 18, 19]
]

test_a_case(test_cases[0])
for i in range(0, 4):
    test_a_case(random_trainer_selection(8, 24))
