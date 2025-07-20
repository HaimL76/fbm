from datetime import datetime

import numpy
import numpy as np
import scipy
from math import sqrt
from pprint import pprint
import matplotlib.pyplot as plt


def cholesky(A):
    size: int = len(A)

    L: list[list[float]] = [[]] * size

    for i in range(size):
        L[i] = [0] * size

    # go over all the columns one by one
    for i in range(size):
        for j in range(i + 1):
            matrix_i = i + 1
            matrix_j = j + 1

            print(f"{matrix_i},{matrix_j}")

            entry_value: float = 0

            if i == j:
                for k in range(j):
                    element: float = L[i][k]

                    element **= 2

                    entry_value += element

                entry_value = sqrt(A[i][j] - entry_value)
            else:
                for k in range(j):
                    element_i: float = L[i][k]
                    element_j: float = L[j][k]

                    element = element_i * element_j

                    entry_value += element

                entry_value = A[i][j] - entry_value

                entry_value *= 1 / L[j][j]

            L[i][j] = entry_value

    return L


def create_covariance_matrix(hurst: float, times: list[float]):
    h: float = 2 * hurst

    size: int = len(times)

    rows: list[list[float]] = [[]] * size

    for i in range(size):
        cols: list[float] = [0] * size

        for j in range(i, size):
            s: float = times[i]
            t: float = times[j]

            first: float = s ** h
            second: float = t ** h
            third: float = abs(s - t) ** h

            val: float = (first + second - third) / 2

            cols[j] = val

        rows[i] = cols

    for j in range(size):
        for i in range(j + 1, size):
            rows[i][j] = rows[j][i]

    return rows


def create_standard_gaussian_vector(n: int):
    now = datetime.now()
    ts = int(now.timestamp())
    np.random.seed(ts)

    standard_gaussian_vector: list[float] = [0] * n

    for index in range(n):
        standard_gaussian_vector[index] = np.random.normal(0, 1, 1)

    return standard_gaussian_vector


def main(n: int, num_paths: int = 1):
    t_range: float = 10

    t_delta: float = t_range / n

    times: list[float] = [t_delta * (i + 1) for i in range(n)]
    rows: list[list[float]] = create_covariance_matrix(0.23, times)

    rows = cholesky(rows)

    L = numpy.array(rows)
    #LT = numpy.transpose(L)

    #A0 = L @ LT

    #pprint(A0)
    u_arr: list = [[]] * num_paths

    for index in range(num_paths):
        standard_gaussian_vector: list[float] = create_standard_gaussian_vector(n)

        v = np.array(standard_gaussian_vector)

        u = L @ v

        u_arr[index] = u

    t_arr = np.array(times)

    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    counter: int = 1

    for u in u_arr:
        ax.plot(t_arr, u)#, label=f"simulation {counter}")

    plt.legend(loc="upper left")
    plt.title("Fractal Brownian Motion Simulation")
    plt.xlabel("Time")
    plt.ylabel("Location")
    plt.show()
    #plt.savefig('fractal_brownian_motion_simulation.png', bbox_inches='tight')
    plt.close(fig=fig)

    _ = 0


main(400, 22)