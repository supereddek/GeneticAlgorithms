import copy
import time

import numpy as np
import random

def tsp_ga(distance_matrix, population_size, crossover_prob, mutation_prob, max_iter, max_time=320):
    # Initialize population
    population = [np.random.permutation(len(distance_matrix)) for _ in range(population_size)]
    population_fitness = [fitness(tour, distance_matrix) for tour in population]

    # Set initial best solution
    best_solution = population[np.argmin(population_fitness)]
    best_fitness = np.min(population_fitness)

    # GA loop
    iterations = 0
    start_time = time.time()
    while True:
        # Select parents
        parents = tournament_selection(population, 2, distance_matrix)
        # Apply crossover and mutation
        offspring = crossover_and_mutation(parents, crossover_prob, mutation_prob)
        # Evaluate offspring
        offspring_fitness = [fitness(tour, distance_matrix) for tour in offspring]
        # Select next generation
        population, population_fitness = next_generation(population, population_fitness, offspring, offspring_fitness)

        # Update best solution
        current_best_index = np.argmin(population_fitness)
        current_best = population[current_best_index]
        if population_fitness[current_best_index] < best_fitness:
            best_solution = current_best
            best_fitness = population_fitness[current_best_index]

        # Check stopping condition
        if stop_condition(iterations, max_iter, start_time, max_time):
            break
        iterations += 1

    return best_solution


def tournament_selection(population, tournament_size, distance_matrix):
    selected = []
    pop = copy.deepcopy(population)
    random.shuffle(pop)
    pairs = [(pop[i], pop[i+1]) for i in range(0, len(pop)-1, 2)]
    for pair in pairs:
        competitors_fitness = [fitness(tour, distance_matrix) for tour in pair]
        winner_index = np.argmin(competitors_fitness)
        selected.append(pair[winner_index])
    return selected


def crossover_and_mutation(parents, crossover_prob, mutation_prob):
    offspring = []
    for i in range(0, len(parents), 2):
        # Crossover
        parent1, parent2 = parents[i], parents[i+1]
        if random.random() < crossover_prob:
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    # Mutation
    for i in range(len(offspring)):
        if random.random() < mutation_prob:
            offspring[i] = mutation(offspring[i])

    return offspring


def crossover(parent1, parent2):
    # Select two random cut points
    cut1, cut2 = random.sample(range(1, len(parent1)), 2)
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    # Create a child by copying the selected segments from parent1
    child1 = np.zeros(len(parent1), dtype=int)
    child1[cut1:cut2] = parent1[cut1:cut2]
    # Fill the remaining positions of the child with the cities from parent2
    next_pos = cut2
    for city in parent2:
        if city not in child1[cut1:cut2]:
            if next_pos == len(child1):
                next_pos = 0
            child1[next_pos] = city
            next_pos += 1
    # Create the second child in a similar way
    child2 = np.zeros(len(parent2), dtype=int)
    child2[cut1:cut2] = parent2[cut1:cut2]
    next_pos = cut2
    for city in parent1:
        if city not in child2[cut1:cut2]:
            if next_pos == len(child2):
                next_pos = 0
            child2[next_pos] = city
            next_pos += 1
    return child1, child2

def fitness(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i]][tour[i+1]]
    total_distance += distance_matrix[tour[-1]][tour[0]]
    return total_distance

def next_generation(population, population_fitness, offspring, offspring_fitness):
    # Combine population and offspring
    combined = list(zip(population + offspring, population_fitness + offspring_fitness))
    # Sort by fitness
    combined = sorted(combined, key=lambda x: x[1])
    # Select best individuals for the next generation
    next_population = [x[0] for x in combined[:len(population)]]
    next_population_fitness = [x[1] for x in combined[:len(population)]]
    return next_population, next_population_fitness

def mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def stop_condition(iterations, max_iterations, start_time, max_time):
    if iterations >= max_iterations:
        print("Maximum number of iterations reached")
        return True
    elif time.time() - start_time > max_time:
        print("Maximum time reached")
        return True
    else:
        return False
