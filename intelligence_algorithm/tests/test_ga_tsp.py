"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
from collections import OrderedDict
from intelligence_algorithm.ga_tsp import TspGaOptimizer

#  Dome, execute part
FIXED_POP = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
# graph A & B
CITY_LIST_SET = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
# CITY_LIST_SET = {'A': 1, 'B': 2, 'C': 3, 'D': 4,
#                  'E': 5, 'F': 6, 'G': 7, 'H': 8,
#                  'I': 9, 'J': 10, 'K': 11, 'L': 12}
CITY_LIST = OrderedDict(sorted(CITY_LIST_SET.items(), key=lambda t: t[1]))

TSP_WEIGHT = [[-1, 3, 6, 7, 2, 1, 6, 4],
              [3, -1, 4, 8, 1, 2, 5, 3],
              [6, 4, -1, 3, 6, 9, 2, 8],
              [7, 8, 3, -1, 7, 2, 5, 6],
              [2, 1, 6, 7, -1, 5, 3, 10],
              [1, 2, 9, 2, 5, -1, 9, 16],
              [6, 5, 2, 5, 3, 9, -1, 6],
              [4, 3, 8, 6, 10, 16, 6, -1]]
# TSP_WEIGHT = [[-1, 5, 7, 9, 11, 13, 15, 13, 11, 9, 7, 5],
#               [5, -1, 5, 7, 9, 11, 13, 15, 13, 11, 9, 7],
#               [7, 5, -1, 5, 7, 9, 11, 13, 15, 13, 11, 9],
#               [9, 7, 5, -1, 5, 7, 9, 11, 13, 15, 13, 11],
#               [11, 9, 7, 5, -1, 5, 7, 9, 11, 13, 15, 13],
#               [13, 11, 9, 7, 5, -1, 5, 7, 9, 11, 13, 15],
#               [15, 13, 11, 9, 7, 5, -1, 5, 7, 9, 11, 13],
#               [13, 15, 13, 11, 9, 7, 5, -1, 5, 7, 9, 11],
#               [11, 13, 15, 13, 11, 9, 7, 5, -1, 5, 7, 9],
#               [9, 11, 13, 15, 13, 11, 9, 7, 5, -1, 5, 7],
#               [7, 9, 11, 13, 15, 13, 11, 9, 7, 5, -1, 5],
#               [5, 7, 9, 11, 13, 15, 13, 11, 9, 7, 5, -1]]


solve = TspGaOptimizer(fixed_pop=FIXED_POP, crossover_rate=CROSSOVER_RATE,
                       mutation_rate=MUTATION_RATE, city_list=CITY_LIST,
                       tsp_weight=TSP_WEIGHT, max_iter=3000).optimize()


