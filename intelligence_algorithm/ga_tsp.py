# coding=utf-8
"""
Author ： g faia
Email : gutianfeigtf@163.com
time : 2016.9.12

file : ga_tsp.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using Genetic Algorithm to solve the TSP.
Computer simulation.
"""
from random import randint, random


class TspGaOptimizer(object):

    def __init__(self, fixed_pop, crossover_rate,
                 mutation_rate, city_list, tsp_weight, max_iter):

        self.fixed_pop = fixed_pop
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter

        if len(city_list) != len(tsp_weight):
            raise Exception("Invalid input.")

        self.city_list = city_list
        self.tsp_weight = tsp_weight
        self.city_num = len(city_list)

    def duplicate_city_list(self):
        """
        Convert the OrderedDict object to list.
        :return: list such as [[], [], ...]
        """
        city_list = []
        temp_list = list(self.city_list.items())

        for tup in temp_list:
            city_list.append(list(tup))

        return city_list

    def generate_code(self):
        """
        Generate a path code, one code is equivalent to a city path.
        :return: a number list.
        """
        city_list = self.duplicate_city_list()
        length = self.city_num
        _code = []

        # Select from city list randomly.
        for i in range(length):
            cur_len = len(city_list)
            select_index = randint(0, cur_len - 1)
            _code.append(city_list[select_index][1])

            # Update the temp city list.
            for j in range(select_index + 1, cur_len):
                city_list[j][1] -= 1
            del city_list[select_index]

        return _code

    def generate_code_list(self):
        """Generate the list contains codes."""
        code_list = []
        fixed_pop = self.fixed_pop

        for i in range(fixed_pop):
            code_list.append(self.generate_code())

        return code_list

    def decode_to_path(self, code):
        """
        Decode the number code to path.
        :param code: the code of path.
        :return: string path.
        """
        city_list = self.duplicate_city_list()
        _path = []

        for c in code:
            _path.append(city_list[c - 1][0])
            del city_list[c - 1]

        return _path

    def generate_path_list(self):
        """Decode the code, and generate the path list."""
        path_list = []
        code_list = self.code_list

        for code in code_list:
            path_list.append(self.decode_to_path(code))

        return path_list

    def select_operator(self):
        """Selection operator, utilize the fitness value to select."""
        def sum_list(lis):
            value = 0
            for ind in range(len(lis)):
                value += lis[ind]
            return value

        def cum_list(lis):
            cum_lis = []
            for ind in range(len(lis)):
                cum_lis.append(sum_list(lis[:(ind + 1)]))
            return cum_lis

        eval_list = []
        path_list = self.path_list
        code_list = self.code_list

        for string in path_list:
            eval_list.append(self.evaluate_path(string))

        sum_value = sum_list(eval_list)
        cum_eval_list = cum_list(eval_list)

        for i in range(len(cum_eval_list)):
            cum_eval_list[i] = cum_eval_list[i] / sum_value

        selected_list = []
        for i in range(len(cum_eval_list)):
            rand_num = random()
            for j in range(len(cum_eval_list)):
                if rand_num < cum_eval_list[j]:
                    selected_list.append(code_list[j])
                    break
                else:
                    pass

        self.code_list = selected_list

    def crossover_operator(self):
        """
        Crossover operator, cross two code.
        In this implement, I choose the half cross in default.
        The crossover probability of crossover operation.
        """

        def crossover(a, b):
            """Crossover operator between two codes."""
            if len(a) == len(b):
                # Generate the random crossover location.
                l = randint(1, len(a) - 1)
                a_front = a[:l]
                a_end = a[l:]
                b_front = b[:l]
                b_end = b[l:]
                a = a_front + b_end
                b = b_front + a_end
                return a, b

        code_len = len(self.code_list)
        crossover_rate = self.crossover_rate
        code_list = self.code_list

        if code_len % 2 == 0:
            pass
        else:
            code_len -= 1
        for i in range(int(code_len / 2)):
            crossover_rand = random()
            if crossover_rand < crossover_rate:
                code_list[2 * i], code_list[2 * i + 1] \
                    = crossover(code_list[2 * i], code_list[2 * i + 1])

    def mutation_operator(self):
        """
        Mutation operator, randomly change a char in string.
        If random number is smaller than mutation rate, Code
        Will generate mutation.
        """
        code_len = self.city_num
        code_list = self.code_list
        mutation_rate = self.mutation_rate

        for code in code_list:
            mutation_rand = random()
            if mutation_rand <= mutation_rate:
                selected_ind = randint(0, code_len - 1)
                selected_list = list(range(1, code_len - selected_ind + 1))
                for i in range(len(selected_list)):
                    if selected_list[i] == code[selected_ind]:
                        del selected_list[i]
                        break
                if selected_list:
                    selected_len = len(selected_list)
                    selected_chr = randint(0, selected_len - 1)
                    code[selected_ind] = selected_list[selected_chr]

    def evaluate_path(self, string):
        """
        Evaluate the eval of path.
        :param string: path string.
        :return: the evaluated value of string.
        """
        seq_list = []
        city_list = self.city_list

        for c in string:
            seq_list.append(city_list[c] - 1)

        length = len(seq_list)
        weight = self.tsp_weight
        value = 0

        for i in range(length):
            front = seq_list[i]
            end = seq_list[(i + 1) % length]
            value += weight[front][end]

        return 1 / value

    def show_best_list(self):
        """
        Show the objects which has best fitness value.
        :return: a list contains best objects which has well fitness.
        """
        eval_list = []
        path_list = self.path_list

        for string in path_list:
            eval_list.append(self.evaluate_path(string))

        best_fitness = 0
        eval_list_len = len(eval_list)
        best_list = []

        for eva in eval_list:
            if eva >= best_fitness:
                best_fitness = eva

        for i in range(eval_list_len):
            if eval_list[i] == best_fitness:
                best_list.append(path_list[i])

        return best_list, 1 / best_fitness

    def update_path_list(self):
        """Update the path list."""
        new_list = []

        for code in self.code_list:
            new_list.append(self.decode_to_path(code))

        self.path_list = new_list

    def optimize(self):
        """"""
        # Initialize the initial population.
        self.code_list = self.generate_code_list()
        self.path_list = self.generate_path_list()

        for i in range(self.max_iter):

            self.select_operator()
            self.crossover_operator()
            self.mutation_operator()
            self.update_path_list()

        return self
