# python implementation of Grey wolf optimization (GWO)
# minimizing Unimodal and Multimodal benchmark functions

import random
import math
import copy  # array-copying convenience
from matplotlib import pyplot as plt
import sys  # max float
import time
import numpy as np
from tqdm import tqdm


# -------fitness functions---------


# sphere function
def fitness_1(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi)
    return fitness_value


# Schwefel 2.22 function
def fitness_2(position):
    fitness_value1 = 0.0
    fitness_value2 = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value1 += abs(xi)
        fitness_value2 *= abs(xi)
    return fitness_value1 + fitness_value2


# Schwefel 1.20 function
def fitness_3(position):
    fitness_value1 = 0
    fitness_value2 = 0
    for i in range(len(position)):
        fitness_value1 = 0
        for j in range(i):
            xj = position[j]
            fitness_value1 += xj
        fitness_value2 += fitness_value1 ** 2
    return fitness_value2


# Schwefel 2.21 function
def fitness_4(position):
    maximum = 0.0
    for i in range(len(position)):
        if abs(position[i]) > maximum:
            maximum = abs(position[i])
    return maximum


# Rosenbrock function
def fitness_5(position):
    fitness_value = 0.0
    for i in range(len(position) - 1):
        xi = position[i]
        xii = position[i + 1]
        fitness_value += (100 * (xii - xi * xi) ** 2 + (xi - 1) ** 2)
    return fitness_value


# Step function
def fitness_6(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi + 0.5) ** 2
    return fitness_value


# Quartic Noise function
def fitness_7(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (i + 1) * (xi ** 4)
    return fitness_value + random.random()


# Schwefel's 2.26 function
def fitness_8(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (-xi) * (math.sin(math.sqrt(abs(xi))))
    return fitness_value


# Rastrigin function
def fitness_9(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value


# Ackley function
def fitness_10(position):
    fitness_value1 = 0.0
    fitness_value2 = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value1 += xi ** 2
        fitness_value2 += math.cos(2 * math.pi * xi)
    fitness_value = -20 * math.exp(-0.2 * math.sqrt((1 / len(position)) * fitness_value1)) - math.exp(
        (1 / len(position)) * fitness_value2) + 20 + math.exp(1)
    return fitness_value


# Griewank function
def fitness_11(position):
    fitness_value1 = 0.0
    fitness_value2 = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value1 += xi ** 2
        fitness_value2 *= math.cos(xi / math.sqrt(i + 1))
    fitness_value = fitness_value1 / 4000 - fitness_value2 + 1
    return fitness_value


# pendlized function
def fitness_12(position):
    fitness_value1 = 0.0
    fitness_value2 = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value1 += Ufun(xi, 10, 100, 4)
    for i in range(len(position) - 1):
        xi = position[i]
        xii = position[i + 1]
        fitness_value2 += ((1 / 4) * (xi + 1) ** 2) * (1 + math.sin(3 * math.pi * (1 + (xii + 1) / 4)) ** 2)
    fitness_value = fitness_value1 + (math.pi / len(position)) * (
            10 * math.sin(math.pi * (1 + (position[0] + 1) / 4)) + fitness_value2 + ((position[-1] + 1) / 4) ** 2)
    return fitness_value


def Ufun(xi, a, k, m):
    o = (k * (xi - a) ** m) * (xi > a) + (k * (-xi - a) ** m) * (xi < -a)
    return o


# -------------------------


# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random()
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness = fitness(self.position)  # curr fitness


# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fitness, dim, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gaama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
    pbar = tqdm(total=max_iter)
    # main loop of gwo
    Iter = 0
    best_iter = []
    best_cost_iter = []
    while Iter < max_iter:
        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
                    2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                    C1 - alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                    C2 - beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                    C3 - gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] += X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness(Xnew)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
        best_iter.append(alpha_wolf.fitness)
        Iter += 1
        pbar.update(1)
    pbar.close()

    return alpha_wolf.position, best_iter


# ----------------------------
# getting input from user
prompt = "Select one of the following functions to optimize:\n" \
         "1) Sphere function.\n2) Schwefel's 2.22 function.\n3) Schwefel's 1.20 function" \
         ".\n4) schwefel's 2.21 function.\n5) Rosenbrock function.\n6) Step function." \
         "\n7) Quartic Noise function.\n8) Schwefel's 2.26 function.\n9) Rastrigin function." \
         "\n10) Ackley function.\n11) Griewank function.\n12) Pendlized function.\n"
userin = input(prompt)

# Driver code for rastrigin function

if userin == '1':
    fitness = fitness_1
    funname = "Sphere function"
    minx = -100
    maxx = 100
elif userin == '2':
    fitness = fitness_1
    funname = "Schwefel's 2.22 function"
    minx = -10
    maxx = 10
elif userin == '3':
    fitness = fitness_3
    funname = "schwefel's 1.20 function"
    minx = -100
    maxx = 100
elif userin == '4':
    fitness = fitness_4
    funname = "Schwefel's 2.21 function"
    minx = -100
    maxx =100
elif userin == '5':
    fitness = fitness_5
    funname = "Rosenbrock function"
    minx = -30
    maxx = 30
elif userin == '6':
    fitness = fitness_6
    funname = "Step function"
    minx = -100
    maxx = 100
elif userin == '7':
    fitness = fitness_7
    funname = "Quartic Noise function"
    minx = -1.28
    maxx = 1.28
elif userin == '8':
    fitness = fitness_8
    funname = "Schwefel's 2.26 function"
    minx = -500
    maxx = 500
elif userin == '9':
    fitness = fitness_9
    funname = "Rastrigin function"
    minx = -5.12
    maxx = 5.12
elif userin == '10':
    fitness = fitness_10
    funname = "Ackley function"
    minx = -32
    maxx = 32
elif userin == '11':
    fitness = fitness_11
    funname = "Griewank function"
    minx = -600
    maxx = 600
elif userin == '12':
    fitness = fitness_12
    funname = "Pendlized function"
    minx = -50
    maxx = 50

dim = int(input("please enter number of variables of the function:\n"))
num_particles = int(input("please enter the number of wolves(between 50 and 200):\n"))
max_iter = int(input("please enter the number of maximum iteration(between 100 and 200):\n"))

print("\nBegin grey wolf optimization on {} function\n".format(funname))
print("Goal is to minimize {} function in " + str(dim) + " variables".format(funname))
print("Function has known min = 0.0 at (", end="")
for i in range(dim - 1):
    print("0, ", end="")
print("0)")


print("Setting num_particles = " + str(num_particles))
print("Setting max_iter = " + str(max_iter))
print("\nStarting GWO algorithm\n")

best_position, best_iter = gwo(fitness, max_iter, num_particles, dim, minx, maxx)

print("\nGWO completed\n")
print("\nBest solution found:")
print(["%.6f" % best_position[k] for k in range(dim)])
err = fitness(best_position)
print("fitness of best solution = %.6f" % err)

print("\nEnd GWO for {}\n".format(funname))
plt.plot(best_iter)
plt.xlabel("Iteration")
plt.ylabel("Cost Function")
print()
print()
plt.show()