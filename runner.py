import os
import statistics
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
import runner_genetic
import scipy.spatial


class args:
    def __init__(self):
        self.sql_host = "localhost"
        self.sql_user = "root"
        self.sql_password = "root"
        self.sql_database = "mnist"


database_clients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

print("total number of participating trainers:", len(database_clients))
model_stats = {}
models = {}
sample_dict = {}
data_dict = {}


def similarities(item, dct):
    sim = []
    for key, data in dct.items():
        m = min(len(item), len(data.y))
        similar_values = 0
        for i in range(m):
            if data.y[i] == item[i]:
                similar_values += 1
        sim.append(round(similar_values / m, 2))
    return sim


def heatmap(dct):
    matrix = []
    for key, data in dct.items():
        simi = similarities(data.y, dct)
        matrix.append(simi)
    print(matrix)


test_data = SQLDataProvider(args()).cache(99999)
print("test data:", test_data.y)
for index, client_idx in enumerate(database_clients):
    data = SQLDataProvider(args()).cache(client_idx)
    model = LogisticRegression(28 * 28, 10)
    trained = train(model, data.batch(8))
    data_dict[client_idx] = data
    model_stats[client_idx] = trained
    models[client_idx] = model
    sample_dict[client_idx] = len(data)
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


def fitness(test_case):
    aggregated = aggregate(dict_select(test_case, model_stats), dict_select(test_case, sample_dict))
    influences = []
    for key in test_case:
        influence = influence_ecl(aggregated, model_stats[key])
        influences.append(influence)
    return statistics.variance(runner_genetic.normalize(influences))


x1 = fitness([10, 1, 0, 11, 9, 7, 8])
x2 = fitness([0, 2, 3, 4, 5, 6, 7, 8])
