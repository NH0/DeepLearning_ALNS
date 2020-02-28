from alns import ALNS, State
from alns.criteria import HillClimbing

import itertools

import numpy.random as rnd

import networkx as nx

import tsplib95
import tsplib95.distances as distances

import matplotlib.pyplot as plt

SEED = 9876

data = tsplib95.load_problem('./data/xqf131.tsp')

solution = tsplib95.load_solution('./data/xqf131.opt.tour')
optimal = data.trace_tours(solution)[0]

print('Total optimal tour length is {0}.'.format(optimal))

def draw_graph(graph, only_nodes=False):
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    func = nx.draw_networkx
    if only_nodes:
        func = nx.draw_networkx_nodes

    func(graph, data.node_coords, node_size=25, with_label=False, ax=ax)
    plt.show()

draw_graph(data.get_graph(), True)

class TspState(State):
    """
    Node : (id, coord)
    Edge : mapping from each node to their only outgoing node
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def copy(self):
        return TspState(self.nodes.copy(), self.edges.copy())

    def objective(self):
        return sum(distances.euclidean(node[1], self.edges[node][1]) for node in self.nodes)

    def to_graph(self):
        graph = nx.Graph()

        for node, coord in self.nodes:
            graph.add_node(node, pos=coord)

        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from[0], node_to[0])

        return graph

degree_of_destruction = 0.25

def edges_to_remove(state):
    return int(len(state.edges) * degree_of_destruction)

def worst_removal(current, random_state):
    """
    Worst removal iteratively removes the 'worst' edges, that is,
    those edges that have the largest distance.
    """
    destroyed = current.copy()

    worst_edges = sorted(destroyed.nodes,
                         key=lambda node: distances.euclidean(node[1],
                                                              destroyed.edges[node][1]))

    for idx in range(edges_to_remove(current)):
        del destroyed.edges[worst_edges[-idx -1]]

    return destroyed

def path_removal(current, random_state):
    """
    Removes an entire consecutive subpath, that is, a series of
    contiguous edges.
    """
    destroyed = current.copy()

    node_idx = random_state.choice(len(destroyed.nodes))
    node = destroyed.nodes[node_idx]

    for _ in range(edges_to_remove(current)):
        node = destroyed.edges.pop(node)

    return destroyed

def random_removal(current, random_state):
    """
    Random removal iteratively removes random edges.
    """
    destroyed = current.copy()

    for idx in random_state.choice(len(destroyed.nodes),
                                   edges_to_remove(current),
                                   replace=False):
        del destroyed.edges[destroyed.nodes[idx]]

    return destroyed

def would_form_subcycle(from_node, to_node, state):
    """
    Ensures the proposed solution would not result in a cycle smaller
    than the entire set of nodes. Notice the offsets: we do not count
    the current node under consideration, as it cannot yet be part of
    a cycle.
    """
    for step in range(1, len(state.nodes)):
        if to_node not in state.edges:
            return False

        to_node = state.edges[to_node]

        if from_node == to_node and step != len(state.nodes) - 1:
            return True

    return False

def greedy_repair(current, random_state):
    """
    Greedily repairs a tour, stitching up nodes that are not departed
    with those not visited.
    """
    visited = set(current.edges.values())

    # This kind of randomness ensures we do not cycle between the same
    # destroy and repair steps every time.
    shuffled_idcs = random_state.permutation(len(current.nodes))
    nodes = [current.nodes[idx] for idx in shuffled_idcs]

    while len(current.edges) != len(current.nodes):
        node = next(node for node in nodes
                    if node not in current.edges)

        # Computes all nodes that have not currently been visited,
        # that is, those that this node might visit. This should
        # not result in a subcycle, as that would violate the TSP
        # constraints.
        unvisited = {other for other in current.nodes
                     if other != node
                     if other not in visited
                     if not would_form_subcycle(node, other, current)}

        # Closest visitable node.
        nearest = min(unvisited,
                      key=lambda other: distances.euclidean(node[1], other[1]))

        current.edges[node] = nearest
        visited.add(nearest)

    return current

random_state = rnd.RandomState(SEED)
state = TspState(list(data.node_coords.items()), {})

initial_solution = greedy_repair(state, random_state)

print("Initial solution objective is {0}.".format(initial_solution.objective()))

draw_graph(initial_solution.to_graph())

alns = ALNS(random_state)

alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(path_removal)
alns.add_destroy_operator(worst_removal)

alns.add_repair_operator(greedy_repair)

# This is perhaps the simplest selection criterion, where we only accept
# progressively better solutions.
criterion = HillClimbing()

result = alns.iterate(initial_solution, [3, 2, 1, 0.5], 0.8, criterion,
                      iterations=5000, collect_stats=True)

solution = result.best_state

objective = solution.objective()

print('Best heuristic objective is {0}.'.format(objective))
print('This is {0:.1f}% worse than the optimal solution, which is {1}.'
      .format(100 * (objective - optimal) / optimal, optimal))

_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax, lw=2)

draw_graph(solution.to_graph())
