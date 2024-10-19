import argparse

import numpy as np

N = 10
GEN = 50
POPULATION_SIZE = 90


class InfluenceGraph:
    def __init__(self, num_nodes):
        self.adj_dict = {i: [] for i in range(num_nodes)}
        self.num_nodes = num_nodes

    def add_edge(self, u, v, probab_campaign1, probab_campaign2):
        self.adj_dict[u].append((v, probab_campaign1, probab_campaign2))


def genetic_evol_algo(graph, initial1, initial2, budget):
    gen0 = init_gen0(graph, budget)
    pass


def init_gen0(graph, budget):
    list(graph.adj_dict.keys())
    population = []

    random_population_size = 2 * POPULATION_SIZE // 9
    good_population_size = 4 * POPULATION_SIZE // 9
    bad_population_size = 3 * POPULATION_SIZE // 9

    # for i in range(random_population_size):
    #     binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
    #     true_cnt = 0
    #     for j in range(len(binary_vector)):
    #         if np.random.random() < np.random.random():
    #             binary_vector[j] = True
    #             true_cnt += 1
    #     population.append(binary_vector)
    #     print(true_cnt)

    for i in range(random_population_size):
        binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
        true_cnt = 0
        l = np.random.randint(0, graph.num_nodes // 3)
        for j in range(l):
            random_node = np.random.randint(0, 2 * graph.num_nodes)
            binary_vector[random_node] = True
            true_cnt += 1
        population.append(binary_vector)
        print(true_cnt)

    for i in range(good_population_size):
        binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
        true_cnt = 0
        j = 0
        while j != budget:
            random_node = np.random.randint(0, 2 * graph.num_nodes)
            if not binary_vector[random_node]:
                binary_vector[random_node] = True
                j += 1
                true_cnt += 1
        population.append(binary_vector)
        print(true_cnt)

    for i in range(bad_population_size):
        binary_vector = np.zeros(2 * graph.num_nodes, dtype=bool)
        true_cnt = 0
        l = np.random.randint(0, budget + graph.num_nodes // 3)
        for j in range(l):
            random_node = np.random.randint(0, 2 * graph.num_nodes)
            binary_vector[random_node] = True
            true_cnt += 1
        population.append(binary_vector)
        print(true_cnt)

    return population


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


def main():
    network_path, initial_seed_path, balanced_seed_path, k = read_user_args()
    graph = read_social_network_dataset(network_path)
    initial1, initial2 = read_seed_dataset(initial_seed_path)
    genetic_evol_algo(graph, initial1, initial2, k)


if __name__ == "__main__":
    main()

'''
    user args instance:
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

'''
