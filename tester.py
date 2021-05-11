from time import perf_counter
import random as rd
import numpy as np
import matplotlib.pyplot as plt

clock = perf_counter



def time_it(function, arg_gen, n, graph = False):
    """
    Compute the average execution time of (function) over (n) executions
    on the arguments given by (arg_gen).
    """
    times = []
    for i in range(n):
        args = arg_gen()
        t0 = clock()
        function(args)
        delta = clock() - t0
        times.append(delta)
    if graph:
        x, y, mean = [], [], times[0]
        for i in range(n-1):
            x.append(i)
            y.append(mean)
            mean *= i
            mean += times[i+1]
            mean /= i+1
        plt.plot(x, y)
        plt.xlabel('Iterations')
        plt.ylabel('Average execution time')
        plt.ylim([np.mean(times) * 0.8, np.mean(times) * 1.2])
        rounded = limit_precision(np.mean(times), 4)
        plt.title('Final average execution time: ' + str(rounded))
        plt.show()
    return np.mean(times)


def limit_precision(f, p):
    """
    Given a float (f), returns (f) with only (p) significant digits.
    """
    power = 0
    while math.floor(f) == 0:
        power += 1
        f *= 10
    f = float(str(f)[:p + 1])
    return f * (10 ** (-power))


def random_hot(size = 20):
    hot = np.zeros((size, ))
    hot[rd.randint(0, size - 1)] = 1
    return hot