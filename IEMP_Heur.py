import argparse
import random
from collections import deque


class InfluenceGraph:
    def __init__(self, num_nodes):
        self.__adj_dict = {i: [] for i in range(num_nodes)}
        self.__num_nodes = num_nodes

    def add_edge(self, u, v, probab_campaign1, probab_campaign2):
        self.__adj_dict[u].append((v, probab_campaign1, probab_campaign2))

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

            for neighbor_tuple in self.__adj_dict[node]:
                neighbor = neighbor_tuple[0]
                probab = neighbor_tuple[campaign]

                if neighbor not in exposed:
                    exposed.add(neighbor)

                if neighbor not in active:
                    if probab >= random.random():
                        active.add(neighbor)
                        queue.append(neighbor)

        return active, exposed

    def diffuse_influence_from_a_node(self, src, active, exposed, campaign):
        active_increment = set()
        exposed_increment = set()

        if src not in active:
            active_increment.add(src)

        if src not in exposed:
            exposed_increment.add(src)

        for neighbor_tuple in self.__adj_dict[src]:
            neighbor = neighbor_tuple[0]
            if neighbor not in exposed:
                exposed_increment.add(neighbor)
            if neighbor not in active:
                probab = neighbor_tuple[campaign]
                if probab >= random.random():
                    active_increment.add(neighbor)

        return active_increment, exposed_increment

    def get_neighbors(self, node):
        return self.__adj_dict[node]

    def get_num_nodes(self):
        return self.__num_nodes

    def __str__(self):
        return str(self.__adj_dict)


def monte_carlo_greedy_heuristic(graph, initial1, initial2, budget, rep):
    num_nodes = graph.get_num_nodes()

    balanced1 = set()
    balanced2 = set()

    while len(balanced1) + len(balanced2) < budget:
        h1_rec = {}
        h2_rec = {}

        for j in range(rep):
            union1 = initial1 | balanced1
            union2 = initial2 | balanced2

            active1, exposed1 = graph.diffuse_influence_bfs(union1, 1)
            active2, exposed2 = graph.diffuse_influence_bfs(union2, 2)
            phi_s1_s2 = compute_phi(num_nodes, exposed1, exposed2)

            for i in range(num_nodes):
                if i not in balanced1 and i not in balanced2:
                    active1_increment, exposed1_increment = graph.diffuse_influence_from_a_node(i, active1, exposed1, 1)
                    active2_increment, exposed2_increment = graph.diffuse_influence_from_a_node(i, active2, exposed2, 2)

                    phi_s1vi_s2 = compute_phi(num_nodes, exposed1 | exposed1_increment, exposed2)
                    phi_s1_s2vi = compute_phi(num_nodes, exposed1, exposed2 | exposed2_increment)

                    h1_rec[i] = h1_rec.get(i, 0) + phi_s1vi_s2 - phi_s1_s2
                    h2_rec[i] = h2_rec.get(i, 0) + phi_s1_s2vi - phi_s1_s2

        new_v1 = None
        new_v2 = None
        new_v1_max_val = -1
        new_v2_max_val = -1

        for j in range(num_nodes):
            if j in h1_rec:
                h1_rec[j] /= rep
                if h1_rec[j] > new_v1_max_val:
                    new_v1_max_val = h1_rec[j]
                    new_v1 = j
            if j in h2_rec:
                h2_rec[j] /= rep
                if h2_rec[j] > new_v2_max_val:
                    new_v2_max_val = h2_rec[j]
                    new_v2 = j

        if new_v1 is not None and (new_v2 is None or h1_rec[new_v1] >= h2_rec[new_v2]):
            balanced1.add(new_v1)
        elif new_v2 is not None:
            balanced2.add(new_v2)

    return balanced1, balanced2


def compute_phi(graph_size, set1, set2):
    return graph_size - len(set1 ^ set2)


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
    balanced1, balanced2 = monte_carlo_greedy_heuristic(graph, initial1, initial2, k, 3)
    write_seed_output(balanced_seed_path, balanced1, balanced2)


if __name__ == "__main__":
    main()

'''
    user args instance:
        map1: 
            terminal: python IEMP_Heur.py -n ./test/Heuristic/map1/dataset1 -i ./test/Heuristic/map1/seed -b ./test/Heuristic/map1/seed_balanced -k 10
            PyCharm: -n ./test/Heuristic/map1/dataset1 -i ./test/Heuristic/map1/seed -b ./test/Heuristic/map1/seed_balanced -k 10
        map2:
            terminal: python IEMP_Heur.py -n ./test/Heuristic/map2/dataset2 -i ./test/Heuristic/map2/seed -b ./test/Heuristic/map2/seed_balanced -k 15
            PyCharm: -n ./test/Heuristic/map2/dataset2 -i ./test/Heuristic/map2/seed -b ./test/Heuristic/map2/seed_balanced -k 15
        map3:
            terminal: python IEMP_Heur.py -n ./test/Heuristic/map3/dataset3 -i ./test/Heuristic/map3/seed -b ./test/Heuristic/map3/seed_balanced -k 15
            PyCharm: -n ./test/Heuristic/map3/dataset3 -i ./test/Heuristic/map3/seed -b ./test/Heuristic/map3/seed_balanced -k 15
        map4:
            terminal: python IEMP_Heur.py -n ./test/Heuristic/map4/dataset4 -i ./test/Heuristic/map4/seed -b ./test/Heuristic/map4/seed_balanced -k 2
            PyCharm: -n ./test/Heuristic/map4/dataset4 -i ./test/Heuristic/map4/seed -b ./test/Heuristic/map4/seed_balanced -k 2

'''
