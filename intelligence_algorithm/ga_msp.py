# coding=utf-8
"""
Author ： g faia
Email : gutianfeigtf@163.com
time : 2017.4.16

file : ga_msp.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using Genetic Algorithm to solve the MSP.
Computer simulation.
"""
from random import randint, random
import csv


class MspPopulation(object):
    """
    The population contains schedule list.
    """
    def __init__(self, graph, fixed_pop, crossover_ratio,
                 mutation_ratio, processor_num, precedence_set):
        # Task_graph is not the DAG. I have converted the DGA artificially
        # To data structure in python.
        # Task_graph such as {'A':{'weight': 3, 'height': 0},...}
        self.task_graph = graph
        self.precedence_set = precedence_set
        self.task_list = sorted(graph.keys())

        # Height_set such as [['A', 'B'], ['C']...].
        # Height_set[height] = subset of height.
        self.height_set = self.generate_height_set()
        self.count_height_set = self.calculate_height_set()
        self.max_height = len(self.height_set) - 1

        # Parameters
        self.fixed_pop = fixed_pop
        self.crossover_rate = crossover_ratio
        self.mutation_ratio = mutation_ratio

        # The number of processors.
        self.processor_num = processor_num
        self.processor_select = []

        for i in range(processor_num):
            self.processor_select.append((i + 1) * (1 / processor_num))

        # Generate the task scheduling list.
        self.schedule_list = self.generate_schedule_list()

    def generate_height_set(self):
        """
        Generate the height set of graph.
        :return: a list such as [['A', 'B'], ['C']...].
        """
        graph = self.task_graph
        max_height = self.get_max_height()
        height_set = []

        for i in range(max_height + 1):
            sub_set = []
            for key in graph.keys():
                if graph[key]['h'] == i:
                    sub_set.append(key)
            height_set.append(sub_set)

        return height_set

    def calculate_height_set(self):
        """
        Calculate the number of tasks in height set.
        :return: count of tasks in height set.
        """
        count_height = []
        height_set = self.height_set

        for lit in height_set:
            length = len(lit)
            count_height.append(length)

        return count_height

    def get_max_height(self):
        """
        :return: the max height of graph.
        """
        max_height = 0
        task_graph = self.task_graph

        for value in task_graph.values():
            if value['h'] > max_height:
                max_height = value['h']

        return max_height

    def generate_schedule_list(self):
        """
        :return: schedule list contains fixed_num of schedules.
        """
        schedule_list = []
        fixed_pop = self.fixed_pop

        for i in range(fixed_pop):
            schedule_list.append(self.generate_schedule())

        return schedule_list

    def generate_schedule(self):
        """
        Randomly generate task schedule.
        :return: For graph, generate a schedule.
        """
        def init_void_schedule(nums):
            void_list = []
            for _ in range(nums):
                void_list.append([])
            return void_list

        processor_num = self.processor_num
        schedule = init_void_schedule(processor_num)
        height_set = self.height_set
        count_set = self.count_height_set

        for i in range(len(height_set)):
            # select the task for processors.
            for j in range(count_set[i]):
                p = self.processor_selector()
                schedule[p].append(height_set[i][j])

        return schedule

    def processor_selector(self):
        """
        :return: which processor have been selected.
        """
        rand_processor = random()
        processor_select = self.processor_select

        for i in range(len(processor_select)):
            if rand_processor < self.processor_select[i]:
                return i

    def select_operator(self):
        """
        Select operator.We need evaluate the fitness of every schedule.
        """
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
        fixed_pop = self.fixed_pop
        schedule_list = self.schedule_list

        for i in range(fixed_pop):
            eval_list.append(1 / self.evaluate_fitness(schedule_list[i]))

        sum_value = sum_list(eval_list)
        cum_eval_list = cum_list(eval_list)

        for i in range(len(cum_eval_list)):
            cum_eval_list[i] = cum_eval_list[i] / sum_value

        selected_list = []

        for i in range(len(cum_eval_list)):
            rand_num = random()
            for j in range(len(cum_eval_list)):
                if rand_num < cum_eval_list[j]:
                    selected_list.append(schedule_list[j])
                    break
                else:
                    pass

        self.schedule_list = selected_list

    def evaluate_fitness(self, schedule):
        """
        Processor_time = [0, 0, ],
        :param: schedule is a list such as [[], []].
        :return: the fitness of the schedule.
        """
        def pre_detect(t, pre_s, task_t):
            pre = pre_s[t]
            for p in pre:
                if p not in task_t.keys():
                    return False
            return True

        graph = self.task_graph
        task_list = self.task_list
        pre_set = self.precedence_set
        processor_num = self.processor_num
        processor_time = []
        task_time = {}

        for i in range(processor_num):
            processor_time.append(0)

        processor_index = 0
        task_index = 0
        task_index_in_processor = []

        for i in range(processor_num):
            task_index_in_processor.append(0)

        while task_index < len(task_list):
            if task_index_in_processor[processor_index] < len(schedule[processor_index]):
                task = schedule[processor_index][task_index_in_processor[processor_index]]
            else:
                processor_index = (processor_index + 1) % processor_num
                continue

            if not pre_detect(task, pre_set, task_time):
                processor_index = (processor_index + 1) % processor_num
            else:
                if not pre_set[task]:
                    task_time[task] = processor_time[processor_index] + graph[task]['w']
                else:
                    max_time = 0
                    for i in pre_set[task]:
                        if task_time[i] > max_time:
                            max_time = task_time[i]
                    if max_time < processor_time[processor_index]:
                        max_time = processor_time[processor_index]
                    task_time[task] = max_time + graph[task]['w']
                task_index_in_processor[processor_index] += 1
                task_index += 1
                processor_time[processor_index] = task_time[task]

        max_value = 0

        for i in task_time.values():
            if i > max_value:
                max_value = i

        return max_value

    def crossover_operator(self):
        """
        Crossover operator.
        In this implement, we use the operator described in
        The "Efficient multiprocessor scheduling Based on genetic algorithms."
        """
        def crossover(a, b):
            max_height = self.max_height
            processor_num = self.processor_num
            rand_height = randint(0, max_height)

            for p in range(processor_num):
                cross_a = a[p]
                cross_b = b[p]
                sep_a = self.find_sep_index(cross_a, rand_height)
                sep_b = self.find_sep_index(cross_b, rand_height)

                cross_a_front = cross_a[:sep_a]
                cross_a_end = cross_a[sep_a:]
                cross_b_front = cross_b[:sep_b]
                cross_b_end = cross_b[sep_b:]
                a[p] = cross_a_front + cross_b_end
                b[p] = cross_b_front + cross_a_end

            return a, b

        list_len = self.fixed_pop
        crossover_rate = self.crossover_rate
        schedule_list = self.schedule_list

        if list_len % 2 == 0:
            pass
        else:
            list_len -= 1

        for i in range(int(list_len / 2)):
            crossover_rand = random()
            if crossover_rand < crossover_rate:
                schedule_list[2 * i], schedule_list[2 * i + 1] \
                    = crossover(schedule_list[2 * i], schedule_list[2 * i + 1])

    def find_sep_index(self, lis, height):
        """
        For height, we find the index which lis[index] > height.
        """
        sep = 0
        task_graph = self.task_graph

        for i in range(len(lis)):
            if task_graph[lis[i]]['h'] <= height and i == (len(lis) - 1):
                sep = len(lis)
            if task_graph[lis[i]]['h'] > height:
                sep = i
                break
        return sep

    def mutation_operator(self):
        """
        Mutation operator.
        In "Genetic Algorithm and Engineering Optimization." p223
        Gen introduce three mutation operators.
        I choose the first operator.
        """
        schedule_list = self.schedule_list
        mutation_ratio = self.mutation_ratio
        max_height = self.max_height

        for schedule in schedule_list:
            mutation_rand = random()
            if mutation_rand <= mutation_ratio:
                random_height = randint(0, max_height)
                # In this application, we only implement the
                # mutation operator for two processors.
                self.mutate_two_processor(schedule, random_height)

    def mutate_two_processor(self, schedule, height):
        """
        Mutation operator for two processors.
        """
        sch_a = schedule[0]
        sch_b = schedule[1]
        sep_a = self.find_sep_index(sch_a, height)
        sep_b = self.find_sep_index(sch_b, height)
        sch_a_front = sch_a[:sep_a]
        sch_a_end = sch_a[sep_a:]
        sch_b_front = sch_b[:sep_b]
        sch_b_end = sch_b[sep_b:]
        schedule[0] = sch_a_front + sch_b_end
        schedule[1] = sch_b_front + sch_a_end

    def show_best_schedule(self):
        """
        Return the best schedule in current schedule list.
        :return: best schedule.
        """
        eval_list = []
        best_schedule_list = []
        schedule_list = self.schedule_list
        fixed_pop = self.fixed_pop

        for schedule in schedule_list:
            eval_list.append(self.evaluate_fitness(schedule))

        max_time = 0

        for i in range(fixed_pop):
            if eval_list[i] >= max_time:
                max_time = eval_list[i]

        for i in range(fixed_pop):
            if eval_list[i] == max_time:
                best_schedule_list.append(schedule_list[i])

        return best_schedule_list, max_time


if __name__ == '__main__':
    # This problem graph in issue
    # "Efficient multiprocessor scheduling
    # Based on genetic algorithms."
    problem_graph = {'A': {'w': 3, 'h': 0},
                     'B': {'w': 3, 'h': 0},
                     'C': {'w': 2, 'h': 1},
                     'D': {'w': 3, 'h': 1},
                     'E': {'w': 2, 'h': 1},
                     'F': {'w': 3, 'h': 2},
                     'G': {'w': 2, 'h': 2},
                     'H': {'w': 1, 'h': 3}}
    precedence = {'A': [],
                  'B': [],
                  'C': ['A'],
                  'D': ['A'],
                  'E': ['B'],
                  'F': ['A', 'C'],
                  'G': ['A', 'B', 'D', 'E'],
                  'H': ['A', 'B', 'C', 'D', 'E', 'F', 'G']}
    # Some parameters.
    fixed_num = 20
    cross = 0.8
    mutation = 0.05
    processor = 2

    population = MspPopulation(problem_graph, fixed_num, cross,
                               mutation, processor, precedence)
    gen = 5000
    csv_list = []

    for ge in range(gen):
        population.select_operator()
        population.crossover_operator()
        population.mutation_operator()
        best_schedule, cost = population.show_best_schedule()

        fitness_value = 1 / cost
        sch_str = []

        for sch in best_schedule:
            pro_str = []
            for pr in sch:
                pro_str.append(''.join(pr))
            sub_str = '.'.join(pro_str)
            if sub_str not in sch_str:
                sch_str.append(sub_str)
        f_string = '|'.join(sch_str)

        csv_list.append([ge + 1, fitness_value, 1 / fitness_value, f_string])

    with open('msp_gen_{0}.csv'.format(gen), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)
