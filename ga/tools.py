import copy
import json
import math
import os
import pprint
import random
import sys
import time

import numpy as np
import psutil
import setproctitle
import torch
from sklearn.cluster import KMeans
from torch import nn

from fedml_api.fedavg.SQLProvider import SQLDataProvider

from fedml_api.model.linear.lr import LogisticRegression


def dict_select(idx, dict_ref):
    new_dict = {}
    for i in idx:
        new_dict[i] = dict_ref[i]
    return new_dict


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def train(model, train_data, epochs=100):
    # change to train mode
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    epoch_loss = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            optimizer.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) > 0:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    weights = model.cpu().state_dict()
    # transform Tensor to list
    return weights


def aggregate(models_dict: dict, sample_dict: dict):
    model_list = []
    training_num = 0

    for idx in models_dict.keys():
        model_list.append((sample_dict[idx], copy.deepcopy(models_dict[idx])))
        training_num += sample_dict[idx]

    # logging.info("################aggregate: %d" % len(model_list))
    (num0, averaged_params) = model_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w

    return averaged_params


def infer(model, test_data):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            pred = model(x)
            loss = criterion(pred, target)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

    return test_acc / test_total, test_loss / test_total


def load(model, stats):
    model.load_state_dict(stats)


random_seed = [0, 1, 2, 3, 4, 5, 6]
random_seed_index = 0


def get_random_seed_index():
    global random_seed_index
    r = random_seed_index
    random_seed_index += 1
    random_seed_index %= len(random_seed)
    return r


def random_trainer_selection(count, clients):
    selected = []
    random.seed(get_random_seed_index())
    while len(selected) < count:
        s = random.randint(0, clients)
        if s not in selected:
            selected.append(s)
    return selected


def influence_ecl(aggregated, model):
    l2_norm = torch.dist(aggregated["linear.weight"], model["linear.weight"], 2)
    return l2_norm.numpy().min()


def normalize(arr):
    total = math.fsum(arr)
    return [i / total for i in arr]


class Clustered:
    def __init__(self, id_label_dict: dict):
        """
        @param id_label_dict dictionary of user id, and the label of this user
        """
        self.id_label_dict = id_label_dict
        self.used_clusters = []
        self.used_models = []

    def reset(self):
        self.used_clusters = []
        self.used_models = []

    def select(self, model_id):
        if model_id in self.used_models:
            return False
        self.used_clusters.append(self.id_label_dict[model_id])
        self.used_models.append(model_id)
        return model_id

    def list(self):
        if len(self.used_models) == len(self.id_label_dict):
            return []
        model_ids = []
        for model_id, label in self.id_label_dict.items():
            if label not in self.used_clusters and model_id not in self.used_models:
                model_ids.append(model_id)
        if len(model_ids) == 0:
            self.used_clusters = []
            return self.list()
        return model_ids

    def __len__(self):
        return len(self.id_label_dict.keys())


def print(txt):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(txt)
