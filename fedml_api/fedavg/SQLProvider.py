import json

import mysql.connector
import torch
from torch import Tensor
import numpy as np


class Sheet:
    def __init__(self, x=None, y=None):
        """
        x is a 2 dimensional array holding the features and their values
        y is a 1 dimensional array holding the label of each feature row
        """
        if y is None:
            y = []
        if x is None:
            x = []
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def extract(self, db, query):
        cursor = db.cursor()
        cursor.execute(query)
        for row in cursor.fetchall():
            self.append(json.loads(row[0]), row[1])
        return self

    def append(self, x, y):
        self.x.append(x)
        self.y.append(y)
        return self

    def to_tensor(self) -> (Tensor, Tensor):
        """
        @param self
        @return: (x:Tensor,labels:Tensor)
        """
        x, y = torch.from_numpy(np.asarray(self.x)).float(), torch.from_numpy(np.asarray(self.y)).long()
        return x, y


class SQLDataProvider:
    def __init__(self, args):
        self.db = mysql.connector.connect(
            host=args.sql_host,
            user=args.sql_user,
            password=args.sql_password,
            database=args.sql_database
        )
        self.x = None
        self.y = None
        self.sheet = None

    def cache(self, client_id, test_data=False):
        # user_id = 'f_' + str(client_id).zfill(5)
        user_id = client_id
        self.sheet = Sheet()
        cursor = self.db.cursor()
        cursor.execute("select data,label from skewed where user_id=%s and is_test = %s", (user_id, test_data))
        for row in cursor.fetchall():
            self.sheet.append(json.loads(row[0]), row[1])
        x, y = self.sheet.to_tensor()
        self.x = x
        self.y = y
        return self

    def size(self):
        return len(self.sheet)

    def batch(self, batch_size):
        if len(self.x) == 0:
            return list()
        batch_data = list()
        batch_size = len(self.x) if batch_size <= 0 or len(self.x) < batch_size else batch_size
        for i in range(0, len(self.x), batch_size):
            batched_x = self.x[i:i + batch_size]
            batched_y = self.y[i:i + batch_size]
            batch_data.append((batched_x, batched_y))
        return batch_data

    def get(self):
        return self.x, self.y

    def is_empty(self):
        return self.sheet is None

    def __len__(self):
        return len(self.x)
