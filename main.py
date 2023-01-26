import numpy as np
from Algorithm import *


def get_data(input_file):
    rows = np.loadtxt(input_file, dtype='i', max_rows=1, unpack=True)
    return np.loadtxt(input_file, dtype='i', skiprows=1, max_rows=int(rows)), rows


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, rows = get_data("gr24.txt")
    for _ in range(10):
        path = tsp_ga(data, 3 * rows, 0.85, 0.06, 100000, 180)
        cost = fitness(path, data)
        print(path.tolist())
        print(cost)

