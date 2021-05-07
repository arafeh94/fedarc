import os
import sys
import statistics

import numpy as np
import psutil
import setproctitle
import torch

import runner_methods
from fedml_api.fedavg.SQLProvider import SQLDataProvider
from runner_genetic import normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from fedml_api.model.linear.lr import LogisticRegression
from runner_methods import *


class args:
    def __init__(self):
        self.sql_host = "localhost"
        self.sql_user = "root"
        self.sql_password = "root"
        self.sql_database = "mnist"


database_clients = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43, 50, 51, 52, 53,
                    60, 61, 62, 63, 70, 71, 72, 73, 80, 81, 82, 83, 90, 91, 92, 93]

print("total number of participating trainers:", len(database_clients))
model_stats = {}
models = {}
sample_dict = {}
data_dict = {}

test_data = SQLDataProvider(args()).cache(20)
# print("test data:", test_data.y)
for index, client_idx in enumerate(database_clients):
    data = SQLDataProvider(args()).cache(client_idx)
    model = LogisticRegression(28 * 28, 10)
    trained = train(model, data.batch(8))
    data_dict[client_idx] = data
    model_stats[client_idx] = trained
    models[client_idx] = model
    sample_dict[client_idx] = len(data)
    test_data = SQLDataProvider(args()).cache(20)
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
    return acc_loss


def fitness(test_case):
    aggregated = aggregate(dict_select(test_case, model_stats), dict_select(test_case, sample_dict))
    influences = []
    for key in test_case:
        influence = influence_ecl(aggregated, model_stats[key])
        influences.append(influence)
    return statistics.variance(normalize(influences))


x0 = fitness([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
x1 = fitness([0, 10, 11, 30, 21, 50, 60, 61, 51, 90])
x2 = fitness([10, 11, 12, 20, 21, 30, 31, 32, 40, 41])
x3 = fitness([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31])
xs = [x1, x2, x3]
for x in xs:
    print(x / x0, x0 < x)
