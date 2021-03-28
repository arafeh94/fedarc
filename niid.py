import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from fedml_core.non_iid.data_distribute import non_iid_partition_with_dirichlet_distribution
import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
)

cursor = mydb.cursor()
cursor.execute("select * from mnist.sample limit 100")
data = cursor.fetchall()
idx = []
labels = []

for row in data:
    idx.append(row[0])
    labels.append(row[3])

