import argparse
from collections import deque

import numpy as np

GEN = 50
POPULATION_SIZE = 90
MUTATION_RATE = 0.2
HALT = 0.98


class InfluenceGraph:
    def __init__(self, num_nodes):
        self.adj_dict = {i: [] for i in range(num_nodes)}
        self.num_nodes = num_nodes

    def add_edge(self, u, v, probab_campaign1, probab_campaign2):
        self.adj_dict[u].append((v, probab_campaign1, probab_campaign2))

    def diffuse_influence_bfs(self, seed, campaign):
        queue = deque()
        active = set()
        exposed = set()

        for node in seed:
            queue.append(node)
            active.add(node)
            exposed.add(node)

        while queue:
            node = queue.popleft()

            for neighbor_tuple in self.adj_dict[node]:
                neighbor = neighbor_tuple[0]
                probab = neighbor_tuple[campaign]

                if neighbor not in exposed:
                    exposed.add(neighbor)

                if neighbor not in active:
                    if probab >= np.random.random():
                        active.add(neighbor)
                        queue.append(neighbor)

        return exposed


class BinaryRepresentation:
    def __init__(self, binary_vector):
        self.binary_vector = binary_vector
        self.fitness_val = 0

    def __str__(self):
        return f"(Fitness Value: {self.fitness_val}, Binary Vector: {self.binary_vector})"


def genetic_evolutionary_algorithm(graph, initial1, initial2, budget):
    gen0, gen0_balanced1, gen0_balanced2 = init_gen0(graph, budget)
    evaluate_fitness(graph, initial1, initial2, gen0_balanced1, gen0_balanced2, gen0, budget)
    gen0.sort(key=lambda binary_representation: binary_representation.fitness_val, reverse=True)
    current_gen = gen0

    best_balanced1, best_balanced2 = convert_binary_representation_to_set_representation(graph, gen0[0])
    best_solution_val = compute_phi(graph, initial1, initial2, best_balanced1, best_balanced2, 150)

    for _ in range(GEN - 1):
        if best_solution_val / graph.num_nodes >= HALT:
            break

        parent_gen = choose_survivors(current_gen)
        next_gen, next_gen_balanced1, next_gen_balanced2 = generate_offspring(graph, parent_gen)
        evaluate_fitness(graph, initial1, initial2, next_gen_balanced1, next_gen_balanced2, next_gen, budget)
        next_gen.sort(key=lambda binary_representation: binary_representation.fitness_val, reverse=True)

        candidate_best_balanced1, candidate_best_balanced2 = convert_binary_representation_to_set_representation(graph,
                                                                                                                 next_gen[
                                                                                                                     0])
        candidate_best_solution_val = compute_phi(graph, initial1, initial2, candidate_best_balanced1,
                                                  candidate_best_balanced2, 150)
        if candidate_best_solution_val > best_solution_val:
            best_balanced1 = candidate_best_balanced1
            best_balanced2 = candidate_best_balanced2
            best_solution_val = candidate_best_solution_val

    return best_balanced1, best_balanced2


def init_gen0(graph, budget):
    population = []
    population_balanced1 = []
    population_balanced2 = []

    random_population_size = 2 * POPULATION_SIZE // 9
    controlled_random_population_size = 4 * POPULATION_SIZE // 9
    ideal_population_size = 3 * POPULATION_SIZE // 9

    for _ in range(random_population_size):
        binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
        balanced1 = set()
        balanced2 = set()
        num_added_nodes = np.random.randint(budget // 3, budget + budget // 2)

        for _ in range(num_added_nodes):
            random_node = np.random.randint(0, 2 * graph.num_nodes)
            binary_vector[random_node] = True
            balanced1.add(random_node) if random_node < graph.num_nodes else balanced2.add(
                random_node % graph.num_nodes)

        population.append(BinaryRepresentation(binary_vector))
        population_balanced1.append(balanced1)
        population_balanced2.append(balanced2)

    for _ in range(controlled_random_population_size):
        binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
        balanced1 = set()
        balanced2 = set()
        num_added_nodes = np.random.randint(budget // 2, budget)

        for _ in range(num_added_nodes):
            random_node = np.random.randint(0, 2 * graph.num_nodes)
            binary_vector[random_node] = True
            balanced1.add(random_node) if random_node < graph.num_nodes else balanced2.add(
                random_node % graph.num_nodes)

        population.append(BinaryRepresentation(binary_vector))
        population_balanced1.append(balanced1)
        population_balanced2.append(balanced2)

    for _ in range(ideal_population_size):
        binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
        balanced1 = set()
        balanced2 = set()
        i = 0

        while i != budget:
            random_node = np.random.randint(0, 2 * graph.num_nodes)
            if not binary_vector[random_node]:
                binary_vector[random_node] = True
                balanced1.add(random_node) if random_node < graph.num_nodes else balanced2.add(
                    random_node % graph.num_nodes)
                i += 1

        population.append(BinaryRepresentation(binary_vector))
        population_balanced1.append(balanced1)
        population_balanced2.append(balanced2)

    return population, population_balanced1, population_balanced2


def evaluate_fitness(graph, initial1, initial2, population_balanced1, population_balanced2, population, budget):
    for i in range(POPULATION_SIZE):
        sign = 1 if len(population_balanced1[i]) + len(population_balanced2[i]) <= budget else -1
        population[i].fitness_val = sign * compute_phi(graph, initial1, initial2, population_balanced1[i],
                                                       population_balanced2[i], 10)


def choose_survivors(population):
    top_20 = population[:20]
    mid_5 = population[40:45]
    bottom_5 = population[85:]
    return top_20 + mid_5 + bottom_5


def generate_offspring(graph, parent_gen):
    next_gen = parent_gen.copy()
    next_gen_balanced1 = []
    next_gen_balanced2 = []
    num_parents = len(parent_gen)

    for node in range(num_parents):
        parent1 = parent_gen[node]

        if node < 10:
            j = node
            while j == node:
                j = np.random.randint(0, 10)
            parent2 = parent_gen[j]
        elif node < num_parents:
            j = node
            while j == node:
                j = np.random.randint(0, num_parents)

        offspring1, offspring2 = apply_two_point_crossover_and_mutate(parent1, parent2)
        next_gen.extend([offspring1, offspring2])

    for binary_representation in next_gen:
        balanced1, balanced2 = convert_binary_representation_to_set_representation(graph, binary_representation)
        next_gen_balanced1.append(balanced1)
        next_gen_balanced2.append(balanced2)

    return next_gen, next_gen_balanced1, next_gen_balanced2


def apply_two_point_crossover_and_mutate(parent1, parent2):
    p1_vector = parent1.binary_vector
    p2_vector = parent2.binary_vector
    length = len(p1_vector)

    point1 = np.random.randint(0, length - 1)
    point2 = np.random.randint(point1 + 1, length)

    offspring1_vector = np.copy(p1_vector)
    offspring2_vector = np.copy(p2_vector)

    offspring1_vector[point1:point2] = p2_vector[point1:point2]
    offspring2_vector[point1:point2] = p1_vector[point1:point2]

    offspring1_vector = apply_flip_bit_mutation(offspring1_vector)
    offspring2_vector = apply_flip_bit_mutation(offspring2_vector)

    offspring1 = BinaryRepresentation(offspring1_vector)
    offspring2 = BinaryRepresentation(offspring2_vector)

    return offspring1, offspring2


def apply_flip_bit_mutation(binary_vector):
    num_bit_flips = np.random.randint(0, len(binary_vector))
    for i in range(num_bit_flips):
        if np.random.random() <= MUTATION_RATE:
            bit_flip = np.random.randint(0, len(binary_vector))
            binary_vector[bit_flip] = not binary_vector[bit_flip]
    return binary_vector


def compute_phi(graph, initial1, initial2, balanced1, balanced2, rep):
    res = 0

    for _ in range(rep):
        exposed1 = graph.diffuse_influence_bfs(initial1 | balanced1, 1)
        exposed2 = graph.diffuse_influence_bfs(initial2 | balanced2, 2)
        phi = graph.num_nodes - len(exposed1 ^ exposed2)
        res += phi

    return res / rep


def convert_binary_representation_to_set_representation(graph, binary_representation):
    binary_vector = binary_representation.binary_vector
    balanced1 = set()
    balanced2 = set()

    for node in range(len(binary_vector)):
        if binary_vector[node]:
            balanced1.add(node) if node < graph.num_nodes else balanced2.add(node % graph.num_nodes)

    return balanced1, balanced2


def read_user_args():
    parser = argparse.ArgumentParser(description="Input args for solving IEMP with a heuristic algorithm")

    parser.add_argument("-n", metavar="<social network>", required=True, dest="network_path",
                        help="Path of the social network file to be read")
    parser.add_argument("-i", metavar="<initial seed set>", required=True, dest="initial_seed_path",
                        help="Path of the initial seed set file to be read")
    parser.add_argument("-b", metavar="<balanced seed set>", required=True, dest="balanced_seed_path",
                        help="Path of the balanced seed set file to be written")
    parser.add_argument("-k", metavar="<budget>", type=int, required=True, dest="k", help="Positive integer budget")

    args = parser.parse_args()
    return args.network_path, args.initial_seed_path, args.balanced_seed_path, args.k


def read_social_network_dataset(file_path):
    with open(file_path, "r") as f:
        num_nodes, num_edges = f.readline().strip().split()
        graph = InfluenceGraph(int(num_nodes))

        for line in f:
            u, v, probab_campaign1, probab_campaign2 = line.strip().split()
            graph.add_edge(int(u), int(v), float(probab_campaign1), float(probab_campaign2))

    return graph


def read_seed_dataset(file_path):
    set1 = set()
    set2 = set()

    with open(file_path, "r") as f:
        set1_size, set2_size = f.readline().strip().split()

        for _ in range(int(set1_size)):
            set1.add(int(f.readline()))

        for _ in range(int(set2_size)):
            set2.add(int(f.readline()))

    return set1, set2


def write_seed_output(file_path, seed1, seed2):
    output_str = f"{len(seed1)} {len(seed2)}\n"
    output_str += "\n".join(str(i) for i in seed1) + "\n"
    output_str += "\n".join(str(i) for i in seed2) + "\n"
    with open(file_path, "w") as f:
        f.write(output_str)


def main():
    network_path, initial_seed_path, balanced_seed_path, k = read_user_args()
    graph = read_social_network_dataset(network_path)
    initial1, initial2 = read_seed_dataset(initial_seed_path)
    balanced1, balanced2 = genetic_evolutionary_algorithm(graph, initial1, initial2, k)
    write_seed_output(balanced_seed_path, balanced1, balanced2)


if __name__ == "__main__":
    main()

'''
    user args instance:
    IEMP_Evol:
        map1: 
            terminal: python IEMP_Evol.py -n ./test/Evolutionary/map1/dataset1 -i ./test/Evolutionary/map1/seed -b ./test/Evolutionary/map1/seed_balanced -k 10
            PyCharm: -n ./test/Evolutionary/map1/dataset1 -i ./test/Evolutionary/map1/seed -b ./test/Evolutionary/map1/seed_balanced -k 10
        map2:
            terminal: python IEMP_Evol.py -n ./test/Evolutionary/map2/dataset2 -i ./test/Evolutionary/map2/seed -b ./test/Evolutionary/map2/seed_balanced -k 15
            PyCharm: -n ./test/Evolutionary/map2/dataset2 -i ./test/Evolutionary/map2/seed -b ./test/Evolutionary/map2/seed_balanced -k 14
        map3:
            terminal: python IEMP_Evol.py -n ./test/Evolutionary/map3/dataset3 -i ./test/Evolutionary/map3/seed -b ./test/Evolutionary/map3/seed_balanced -k 15
            PyCharm: -n ./test/Evolutionary/map3/dataset3 -i ./test/Evolutionary/map3/seed -b ./test/Evolutionary/map3/seed_balanced -k 14
        map4:
            terminal: python IEMP_Evol.py -n ./test/Evolutionary/map4/dataset4 -i ./test/Evolutionary/map4/seed -b ./test/Evolutionary/map4/seed_balanced -k 2
            PyCharm: -n ./test/Evolutionary/map4/dataset4 -i ./test/Evolutionary/map4/seed -b ./test/Evolutionary/map4/seed_balanced -k 2
    Evaluator:
        map1:
            PyCharm: -n ./test/Evolutionary/map1/dataset1 -i ./test/Evolutionary/map1/seed -b ./test/Evolutionary/map1/seed_balanced -k 10 -o ./test/Evolutionary/map1/output
        map2:
            PyCharm: -n ./test/Evolutionary/map2/dataset2 -i ./test/Evolutionary/map2/seed -b ./test/Evolutionary/map2/seed_balanced -k 14 -o ./test/Evolutionary/map2/output

'''
