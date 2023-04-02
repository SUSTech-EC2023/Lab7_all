import numpy as np
import matplotlib.pyplot as plt

# Game settings
R = 3
S = 0
T = 5
P = 1

def play_2IPD(player1_move, player2_move):
    if player1_move and player2_move:
        return R, R
    elif not player1_move and player2_move:
        return S, T
    elif player1_move and not player2_move:
        return T, S
    else:
        return P, P

def encode_memory(memory):
    return sum(2**i * move for i, move in enumerate(memory))

def decode_memory(encoded_memory, memory_size):
    return [(encoded_memory >> i) & 1 for i in range(memory_size)]

def create_initial_population(pop_size, memory_size):
    return np.random.randint(0, 2, size=(pop_size, 2 ** memory_size))

def calculate_fitness(population1, population2, memory_size, num_rounds):
    fitness = np.zeros(len(population1))

    for i, player1_strategy in enumerate(population1):
        for j, player2_strategy in enumerate(population2):
            player1_memory = [0] * memory_size
            player2_memory = [0] * memory_size

            for round in range(num_rounds):
                player1_move = player1_strategy[encode_memory(player1_memory)]
                player2_move = player2_strategy[encode_memory(player2_memory)]

                player1_score, player2_score = play_2IPD(player1_move, player2_move)
                fitness[i] += player1_score

                player1_memory.pop(0)
                player1_memory.append(player2_move)

                player2_memory.pop(0)
                player2_memory.append(player1_move)

    return fitness

def mutation(parent, mutation_rate):
    return np.where(np.random.rand(*parent.shape) < mutation_rate, 1 - parent, parent)

def tournament_selection(population, fitness, tournament_size):
    indices = np.random.choice(len(population), tournament_size)
    best_index = indices[np.argmax(fitness[indices])]
    return population[best_index]

def evolve_population(population, fitness, mutation_rate, tournament_size, elitism_rate):
    offspring_population = []
    
    # Elitism
    num_elites = int(len(population) * elitism_rate)
    elite_indices = np.argsort(fitness)[-num_elites:]
    offspring_population.extend(population[elite_indices])

    for _ in range(len(population) - num_elites):
        parent1 = tournament_selection(population, fitness, tournament_size)
        # parent2 = tournament_selection(population, fitness, tournament_size)

        offspring = np.where(np.random.rand(*parent1.shape) < mutation_rate, 1 - parent1, parent1)
        offspring_population.append(offspring)

    return np.array(offspring_population)

def run_co_evolutionary_algorithm(num_generations, pop_size, mutation_rate, memory_size, num_rounds, tournament_size, elitism_rate):
    player1_population = create_initial_population(pop_size, memory_size)
    player2_population = create_initial_population(pop_size, memory_size)

    player1_scores = []
    player2_scores = []

    for generation in range(num_generations):
        player1_fitness = calculate_fitness(player1_population, player2_population, memory_size, num_rounds)
        player2_fitness = calculate_fitness(player2_population, player1_population, memory_size, num_rounds)

        player1_scores.append(np.mean(player1_fitness))
        player2_scores.append(np.mean(player2_fitness))

        player1_population = evolve_population(player1_population, player1_fitness, mutation_rate, tournament_size, elitism_rate)
        player2_population = evolve_population(player2_population, player2_fitness, mutation_rate, tournament_size, elitism_rate)

    plt.plot(player1_scores, label="Player 1")
    plt.plot(player2_scores, label="Player 2")
    plt.xlabel("Generation")
    plt.ylabel("Average Score")
    plt.legend()
    plt.show()

    return player1_population, player2_population


if __name__ == "__main__":
    # Parameters
    num_generations = 100
    pop_size = 50
    mutation_rate = 0.1
    memory_size = 3
    num_rounds = 50
    tournament_size = 5
    elitism_rate = 0.1

    # Run co-evolutionary algorithm
    print("Running co-evolutionary algorithm...")
    player1_population, player2_population = run_co_evolutionary_algorithm(num_generations, pop_size, mutation_rate, memory_size, num_rounds, tournament_size, elitism_rate)

    # Print final populations
    print("Player 1 final population:")
    print(player1_population)

    print("Player 2 final population:")
    print(player2_population)
