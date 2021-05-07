import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))
from ga.context import Context

clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

context = Context(clients)
for i in range(5):
    context.build(round_idx=i)
    context.test_selection_accuracy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'one label selection accuracy')
