import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))
from ga.context import Context

clients = [
    0, 1, 2, 3, 4, 5, 6,  # label 0 clients
    10, 11, 12, 13, 14,  # label 1 clients
    20, 21, 22, 23, 24,  # label 2 clients
    30, 31, 32, 33, 34,  # label 3 clients
    40, 41, 42, 43, 44,  # label 4 clients
    50, 51, 52, 53, 54,  # label 5 clients
    60, 61, 62, 63, 64,  # label 6 clients
    70, 71, 72, 73, 74,  # label 7 clients
    80, 81, 82, 83, 84,  # label 8 clients
    90, 91, 92, 93, 94,  # label 9 clients
    101,  # client that have 80 rows labeled 0
    102, 103, 104, 105,  # each have between 1 and 10 rows
    106, 107, 108, 109, 110  # each have 1 and 10 rows
]

context = Context(clients)
context.build(True)
context.test_selection_accuracy([0, 1, 2, 3, 4, 5, 6], 'one label selection accuracy')
context.test_selection_accuracy([0, 1, 2, 10, 20, 21, 30, 31, 40, 41], '4 out of 10 labels selection accuracy')
context.test_selection_accuracy([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], '10 out of 10 labels selection accuracy')
context.test_selection_accuracy([101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                                '10 out of 10 skewed labels selection accuracy')
