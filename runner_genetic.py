import math
import statistics

from numpy.random import default_rng
import random
import numpy as np


def select_random(genes, size):
    rng = default_rng()
    numbers = rng.choice(len(genes), size=size, replace=False)
    return [genes[i] for i in numbers]


def build_population(genes, p_size, c_size):
    population = []
    for i in range(p_size):
        population.append(select_random(genes, c_size))
    return population


def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if random.uniform(0, 1) < r_cross:
        pt = np.random.randint(1, len(p1) - 1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(arr, genes, r_mut):
    copy = arr.copy()
    for index, value in enumerate(copy):
        if random.uniform(0, 1) < r_mut:
            copy[index] = genes[random.randint(0, len(genes) - 1)]
    return copy


def selection(population, scores):
    selected = []
    selected_indexes = []
    prob = wheel(population, scores)
    while len(selected) < len(population) / 2:
        for index, item in enumerate(population):
            if prob[index] > random.uniform(0, 1) and index not in selected_indexes:
                selected.append(item)
                selected_indexes.append(index)
                if len(selected) >= len(population) / 2:
                    break
    return selected


def populate(population, p_size):
    copy = population.copy()
    while len(copy) < p_size:
        p1 = np.random.randint(0, len(population))
        p2 = np.random.randint(0, len(population))
        pn = crossover(population[p1], population[p2], 1)
        copy += pn
    while len(copy) > p_size:
        copy.pop()
    return copy


def wheel(population, scores):
    total = np.sum(scores)
    return [scores[index] / total for index, item in enumerate(population)]


def normalize(arr):
    total = math.fsum(arr)
    return [i / total for i in arr]


def duplicate(arr):
    return len([x for x in arr if arr.count(x) > 1]) > 1


def clean(population):
    temp = []
    for index, item in enumerate(population):
        if not duplicate(item):
            temp.append(item)
    if len(temp) % 2 != 0:
        temp.pop()
    for i in temp:
        if duplicate(i):
            clean(population)
            print("sa")
    return temp


def ga(fitness, genes, desired, max_iter, r_cross, r_mut, p_size, c_size):
    population = build_population(genes, p_size, c_size)
    minimize = 99999999999
    n_iter = 0
    while n_iter < max_iter and minimize > desired:
        scores = [fitness(chromosome) for chromosome in population]
        for index, ch in enumerate(population):
            if scores[index] < minimize:
                minimize = scores[index]
                print(ch, minimize)
        population = selection(population, scores)
        population = populate(population, int(p_size * 3 / 4))
        population += build_population(genes, int(p_size / 4), 8)
        children = list()
        for i in range(0, len(population), 2):
            p1, p2 = population[i], population[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, genes, r_mut)
                children.append(c)
        population = clean(children)
    pass


def fitness(arr):
    return statistics.variance(normalize(arr))


