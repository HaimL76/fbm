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
    standard_gaussian_vector: list[float] = [0] * n

    for index in range(n):
        standard_gaussian_vector[index] = np.random.normal(0, 0.0001, 1)

    return standard_gaussian_vector


def calculate_fbm(hurst: float, n: int, num_paths: int = 1, cholesky_rows = None):
    t_range: float = 10

    t_delta: float = t_range / n

    times: list[float] = [t_delta * (i + 1) for i in range(n)]
    rows: list[list[float]] = create_covariance_matrix(hurst=hurst, times=times)

    if not isinstance(cholesky_rows, list) or len(cholesky_rows) != n:
        cholesky_rows: list[list[float]] = cholesky(rows)

    L = numpy.array(cholesky_rows)
    #LT = numpy.transpose(L)

    #A0 = L @ LT

    #pprint(A0)
    u_arr: list = [[]] * num_paths

    current_timestamp = None

    for index in range(num_paths):
        now = datetime.now()
        ts = int(now.timestamp())

        if current_timestamp != ts:
            current_timestamp = ts
            np.random.seed(ts)

        standard_gaussian_vector: list[float] = create_standard_gaussian_vector(n)

        v = np.array(standard_gaussian_vector)

        u = L @ v

        u_arr[index] = u

        print(f"[{index}] {u[0]}")

    msd: list[float] = [0] * len(times)

    index = 0

    h: float = hurst * 2

    for index in range(1, len(times)):
        t: float = times[index]

        t_msd_sum: float = 0

        for u in u_arr:
            u0: float = u[0]
            ui: float = u[index]

            d: float = abs(ui - u0)

            d **= 2

            t_msd_sum += d

        t_msd = t_msd_sum / len(u)
        th: float = t ** h
        ratio: float = t_msd / th
        print(f"[{index}] t={t}, t_msd={t_msd}, th={th}, ratio={ratio}")
        msd[index] = t_msd

    t_arr = np.array(times)

    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    counter: int = 0

    for u in u_arr:
        counter += 1
        ax.plot(t_arr, u, label=f"path {counter}")

    if num_paths < 10:
        plt.legend(loc="upper left")
    
    plt.title("Fractal Brownian Motion Simulation")
    plt.xlabel("Time")
    plt.ylabel("Location")
    #plt.show()
    plt.savefig(f'fractal_brownian_motion_simulation_{hurst}.png', bbox_inches='tight')
    plt.close(fig=fig)

    return cholesky_rows

cholesky_rows = None

cholesky_rows = calculate_fbm(hurst=0.5, n=1000, num_paths=25000)
cholesky_rows = calculate_fbm(hurst=0.25, n=999, num_paths=25000)
cholesky_rows = calculate_fbm(hurst=0.75, n=999, num_paths=25000)