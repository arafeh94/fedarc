from runner_methods import Clustered

selector = Clustered({0: 1, 1: 1})

print(selector.list())
print(selector.select(0))
print(selector.list())
print(selector.select(1))
print(selector.list())

clustering = Clustered({
    10: 1, 11: 1, 12: 1, 13: 2, 14: 2, 15: 2, 16: 3, 17: 3, 18: 3, 19: 4, 20: 4, 21: 4, 22: 5, 23: 5, 24: 5,
    25: 6, 26: 6, 27: 6, 28: 7, 29: 7,
})
