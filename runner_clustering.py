import os
import sys

from sklearn.cluster import DBSCAN, KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
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
labels = {}

test_data = SQLDataProvider(args()).cache(20)
# print("test data:", test_data.y)
for index, client_idx in enumerate(database_clients):
    data = SQLDataProvider(args()).cache(client_idx)
    model = LogisticRegression(28 * 28, 10)
    trained = train(model, data.batch(8))
    data_dict[client_idx] = data
    model_stats[client_idx] = trained
    models[client_idx] = model
    labels[client_idx] = data.y
    sample_dict[client_idx] = len(data)
    test_data = SQLDataProvider(args()).cache(20)
    print("model accuracy:", infer(model, test_data.batch(8)))


def extract():
    weights = []
    for idx, model in models.items():
        state = model.state_dict()
        weights.append(state['linear.weight'].numpy().flatten())
    return weights


X = extract()
kmeans = KMeans(n_clusters=10).fit(X)
print(kmeans.labels_)
print(labels)
