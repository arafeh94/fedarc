import json
import os
import statistics
import sys

import mysql.connector
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

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='mnist'
)

cursor = db.cursor()
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

client_id = 0
for label in labels:
    print("working for label: " + str(label))
    cursor.execute("select data,label from sample where label = " + str(label))
    count = 0
    skip = 0
    for row in cursor.fetchall():

        if count < 20:
            values = (client_id, row[0], row[1], 0)
            query = "insert into skewed (id, user_id, data, label, is_test) values (null,%s,%s,%s,%s)"
            db.cursor().execute(query, values)
            count += 1
        else:
            count = 0
            client_id += 1
        if count == 0 and client_id % 10 == 0:
            break
db.commit()
