"""
Author ： g faia
Email : gutianfeigtf@163.com
time : 2017.4.2

file : simple_sort.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple_sort contains some sort methods.
"""
import generate_data


def create_one_two_three_list(count):
    """generate the list containing random number in [1,2,3],
    store in number.
    """
    random_list = generate_data.generate_random_float_list(count)
    number_list = []
    constants = [1/3, 2/3, 1]

    for r in random_list:
        for c in constants:
            if r < c:
                number_list.append(int(c * 3))

    return number_list


def netherlands_flag(sort_array):
    """iterate the list, statistic the number of 1 2 3,
    then generate the list sorted one by one.
    """
    sorted_list = []
    length = len(sort_array)
    count = [0, 0, 0]

    for i in range(length):
        if sort_array[i] == 1:
            count[0] += 1
        elif sort_array[i] == 2:
            count[1] += 1
        elif sort_array[i] == 3:
            count[2] += 1

    for i in range(3):
        for j in range(count[i]):
            sorted_list.append(i+1)
    return sorted_list


def netherlands_flag_other(sort_array):
    """other method to solve the problem
    """
    sorted_list = []
    begin = current = 0
    length = len(sort_array)
    end = length - 1

    for i in range(length):
        sorted_list.append(sort_array[i])

    while current <= end:
        if sorted_list[current] == 1:
            swap(sorted_list, current, begin)
            begin += 1
            current += 1
        elif sorted_list[current] == 2:
            current += 1
        elif sorted_list[current] == 3:
            swap(sorted_list, current, end)
            end -= 1
    return sorted_list


def swap(lists, first, second):
    """swap two element in list.

    lists, type list
    first second, index
    """
    temp = lists[first]
    lists[first] = lists[second]
    lists[second] = temp


def bubble_sort(sort_array):
    """Simple bubble sort algorithm.

    sort_array, type list
    """
    length = len(sort_array)
    if length <= 1:
        return sort_array
    else:
        for i in range(length):
            for j in range(length - i - 1):
                if sort_array[j] > sort_array[j + 1]:
                    swap(sort_array, j, j + 1)
    return sort_array


def insert_sort(sort_array):
    """Insertion sort algorithm.

    sort_array, type list
    """
    length = len(sort_array)
    if length <= 1:
        return sort_array
    else:
        for i in range(1, length):
            temp = sort_array[i]
            j = i - 1
            while j >= 0:
                if temp < sort_array[j]:
                    sort_array[j + 1] = sort_array[j]
                else:
                    break
                j -= 1
            sort_array[j + 1] = temp
    return sort_array


def selection_sort(sort_array):
    """Simple selection sort algorithm

    sort_array, type list
    """
    length = len(sort_array)
    if length <= 1:
        return sort_array
    else:
        for i in range(length):
            j = i
            min_index = j
            while j < length:
                if sort_array[j] < sort_array[min_index]:
                    min_index = j
                j += 1
            swap(sort_array, min_index, i)
    return sort_array


if __name__ == "__main__":
    l = generate_data.generate_random_integer_list()
    print(l)
