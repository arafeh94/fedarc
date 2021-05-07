import math
import os
import sys
import statistics

from sklearn.cluster import KMeans

from fedml_api.fedavg.SQLProvider import SQLDataProvider
from fedml_api.model.linear.lr import LogisticRegression
import ga.tools as tools


class args:
    def __init__(self):
        self.sql_host = "localhost"
        self.sql_user = "root"
        self.sql_password = "root"
        self.sql_database = "mnist"


class Context:
    def __init__(self, all_clients):
        self.args = args()
        self.all_clients = all_clients
        self.model_stats = {}
        self.models = {}
        self.sample_dict = {}
        self.data_dict = {}
        self.test_data = SQLDataProvider(args()).cache(100)

    def build(self, test_models=False):
        print("Building Models --Started")
        for index, client_idx in enumerate(self.all_clients):
            data = SQLDataProvider(args()).cache(client_idx)
            model = LogisticRegression(28 * 28, 10)
            trained = tools.train(model, data.batch(8))
            self.data_dict[client_idx] = data
            self.model_stats[client_idx] = trained
            self.models[client_idx] = model
            self.sample_dict[client_idx] = len(data)
            if test_models:
                print("model accuracy:", tools.infer(model, self.test_data.batch(8)))
        print("Building Models --Finished")

    def cluster(self, cluster_size=10):
        print("Clustering Models --Started")
        weights = []
        client_ids = []
        clustered = {}
        for client_id, stats in self.model_stats.items():
            client_ids.append(client_id)
            weights.append(stats['linear.weight'].numpy().flatten())
        kmeans = KMeans(n_clusters=cluster_size).fit(weights)
        for i, label in enumerate(kmeans.labels_):
            clustered[client_ids[i]] = label
        print("Clustering Models --Finished")
        return clustered

    def test_selection_accuracy(self, client_idx, title='test accuracy', output=True):
        print('-----------------' + title + '-----------------')
        global_model_stats = tools.aggregate(tools.dict_select(client_idx, self.model_stats),
                                             tools.dict_select(client_idx, self.sample_dict))
        global_model = LogisticRegression(28 * 28, 10)
        tools.load(global_model, global_model_stats)
        acc_loss = tools.infer(global_model, self.test_data.batch(8))
        if output:
            print("test case:", client_idx)
            print("global model accuracy:", acc_loss[0], 'loss:', acc_loss[1])
        return acc_loss

    def test_selection_fitness(self, client_idx, title='test_fitness', output=True):
        aggregated = tools.aggregate(tools.dict_select(client_idx, self.model_stats),
                                     tools.dict_select(client_idx, self.sample_dict))
        influences = []
        for key in client_idx:
            influence = tools.influence_ecl(aggregated, self.model_stats[key])
            influences.append(influence)
        fitness = statistics.variance(tools.normalize(influences))
        fitness = fitness * 10 ** 5
        if output:
            print("test case:", client_idx)
            print("selection fitness:", fitness)
        return fitness

    def fitness(self, client_idx):
        return self.test_selection_fitness(client_idx, output=False)
