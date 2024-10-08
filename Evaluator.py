import argparse
import random
from collections import deque


class InfluenceGraph:
    def __init__(self, num_nodes):
        self.__adj_dict = {i: [] for i in range(num_nodes)}
        self.__num_nodes = num_nodes

    def add_edge(self, u, v, probab_campaign1, probab_campaign2):
        self.__adj_dict[u].append((v, probab_campaign1, probab_campaign2))

    def get_neighbors(self, node):
        return self.__adj_dict[node]

    def get_num_nodes(self):
        return self.__num_nodes

    def __str__(self):
        return str(self.__adj_dict)


def simulate_monte_carlo(rep, graph, initial1, initial2, balanced1, balanced2):
    res = 0

    for _ in range(rep):
        exposed1 = diffuse_influence(graph, initial1 | balanced1, 1)
        exposed2 = diffuse_influence(graph, initial2 | balanced2, 2)
        res += graph.get_num_nodes() - len(exposed1 ^ exposed2)

    return res / rep


def diffuse_influence(graph, seed, campaign):
    queue = deque()
    active = set()
    exposed = set()

    for node in seed:
        queue.append(node)
        active.add(node)
        exposed.add(node)

    while queue:
        node = queue.popleft()

        for neighbor_tuple in graph.get_neighbors(node):
            neighbor = neighbor_tuple[0]
            probab = neighbor_tuple[campaign]

            if neighbor not in exposed:
                exposed.add(neighbor)

            if neighbor not in active:
                if probab >= random.random():
                    active.add(neighbor)
                    queue.append(neighbor)

    return exposed


def read_user_args():
    parser = argparse.ArgumentParser(description="Evaluator for IEMP")

    parser.add_argument("-n", metavar="<social network>", required=True, dest="network_path",
                        help="Path of the social network file")
    parser.add_argument("-i", metavar="<initial seed set>", required=True, dest="initial_seed_path",
                        help="Path of the initial seed set file")
    parser.add_argument("-b", metavar="<balanced seed set>", required=True, dest="balanced_seed_path",
                        help="Path of the balanced seed set file")
    parser.add_argument("-k", metavar="<budget>", type=int, required=True, dest="k", help="Positive integer budget")
    parser.add_argument("-o", metavar="<objective value output path>", dest="output_path", required=True,
                        help="Path of the objective value output")

    args = parser.parse_args()
    return args.network_path, args.initial_seed_path, args.balanced_seed_path, args.k, args.output_path


def read_social_network_dataset(file_path):
    with open(file_path, "r") as f:
        num_nodes, num_edges = f.readline().strip().split()
        graph = InfluenceGraph(int(num_nodes))

        for line in f:
            u, v, probab_campaign1, probab_campaign2 = line.strip().split()
            graph.add_edge(int(u), int(v), float(probab_campaign1), float(probab_campaign2))

    return graph


def read_seed_dataset(file_path):
    set1, set2 = set(), set()

    with open(file_path, "r") as f:
        set1_size, set2_size = f.readline().strip().split()

        for _ in range(int(set1_size)):
            set1.add(int(f.readline().strip()))

        for _ in range(int(set2_size)):
            set2.add(int(f.readline().strip()))

    return set1, set2


def write_output(file_path, output_data):
    with open(file_path, "w") as f:
        f.write(str(output_data))


def solve():
    network_path, initial_seed_path, balanced_seed_path, k, output_path = read_user_args()
    graph = read_social_network_dataset(network_path)
    initial1, initial2 = read_seed_dataset(initial_seed_path)
    balanced1, balanced2 = read_seed_dataset(balanced_seed_path)
    obj_val = simulate_monte_carlo(1000, graph, initial1, initial2, balanced1, balanced2)
    write_output(output_path, obj_val)


if __name__ == "__main__":
    solve()

'''
    user args instance:
        map1: 
            terminal: python Evaluator.py -n ./test/Evaluator/map1/dataset1 -i ./test/Evaluator/map1/seed -b ./test/Evaluator/map1/seed_balanced -k 15 -o ./test/Evaluator/map1/output
            PyCharm: -n ./test/Evaluator/map1/dataset1 -i ./test/Evaluator/map1/seed -b ./test/Evaluator/map1/seed_balanced -k 15 -o ./test/Evaluator/map1/output
        map2:
            terminal: python Evaluator.py -n ./test/Evaluator/map2/dataset2 -i ./test/Evaluator/map2/seed -b ./test/Evaluator/map2/seed_balanced -k 15 -o ./test/Evaluator/map2/output
            PyCharm: -n ./test/Evaluator/map2/dataset2 -i ./test/Evaluator/map2/seed -b ./test/Evaluator/map2/seed_balanced -k 15 -o ./test/Evaluator/map2/output
        
'''
