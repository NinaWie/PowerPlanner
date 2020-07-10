import numpy as np
import queue
import math
from heapq import heappush, heappop


class Label():

    def __init__(self, node, cost, angle):
        self.node = node
        self.cost = cost
        self.angle = angle

    def unwrap(self):
        return self.node, self.cost, self.angle

    def __lt__(self, label):
        _, c, ang = label.unwrap()
        return self.rec_compare(self.cost, c, self.angle, ang)

    def rec_compare(self, a, b, a_angle, b_angle):
        if not a:
            return a_angle != b_angle
            # normalerweise True, aber für uns ist wichtig dass sie verschieden sind wenn der Winkel anders ist... I think
        if a[0] < b[0]:
            return True
        elif a[0] > b[0]:
            return False
        else:
            return self.rec_compare(a[1:], b[1:], a_angle, b_angle)


class Graph():

    def __init__(self, nodes, edges, d):
        """
        Creates a graph from a list of nodes and a list of edges with d-dimensional cost vectors.
        Nodes and edges are tuples.
        node = (name, x, y) where x and y are coordinates of the node location in 2d space
        edge = (from, to, cost) where from and to are nodes and cost is a vector
                The last value in the cost vector resembles the angle cost of a path.
        """
        self.d = d

        # assigning IDs to nodes
        self.nodes = []
        self.name_to_num = dict()
        self.num_to_name = dict()

        for idx, (name, x, y) in enumerate(nodes):
            self.name_to_num[name] = idx
            self.num_to_name[idx] = name
            self.nodes.append((x, y))

        # creating matrix representation
        # every entry is of the form (from, to, values), where
        # values = (is_edge, cost, out_angle, in_angle),
        # so the first element is binary, the following d elements are the cost vector
        # and the last elements are the angles
        self.edge_matrix = np.zeros((len(nodes), len(nodes), d + 3))

        for (f, t, c) in edges:
            # getting index
            i = self.name_to_num[f]
            j = self.name_to_num[t]

            # storing that there is an edge between the two
            self.edge_matrix[i, j, 0] = 1

            # storing cost
            self.edge_matrix[i, j, 1:d + 1] = c

            # calculating and storing angle
            fx, fy = self.nodes[i]
            tx, ty = self.nodes[j]
            out_angle, in_angle = self.calculate_angles((fx, fy), (tx, ty))
            self.edge_matrix[i, j, d + 1] = out_angle
            self.edge_matrix[i, j, d + 2] = in_angle

    def calculate_angles(self, f, t):
        """
        Calculates the incoming and outgoing angles of a node.
        If none of this works, the mistake probably lies here because I suck at math :(
        """
        f = np.array(f)
        t = np.array(t)
        vector = t - f
        base = [0, 1]  # vector that points straight upwards is base
        out_angle = math.acos(
            math.radians(
                np.dot(vector, base) / np.linalg.norm(vector) *
                np.linalg.norm(base)
            )
        )
        in_angle = (180 + out_angle) % 360

        return out_angle, in_angle

    def dominance_check(self, labels, cost, new_angle):
        """
        returns true if cost is dominated in labels.

        This is where the theoretical runtime collapses!
        It is possible to do these dominance checks in polynomial time, but this is a naive
        implementation that explodes with growing cost vector dimensionality.

        (Also, this feels needlessly complex but I'm too lazy to clean it up)
        """
        for (label, angle) in labels:
            for i in range(self.d):
                if cost[i] < label[i]:
                    break
            else:  # we got through that entire vector without ever being smaller
                # if the vectors are the same, in which case the sums are the same, we
                # want to return True only if the angles are the same as well
                if np.sum(label) == np.sum(cost):
                    if angle == new_angle:
                        return True
                    else:
                        return False  # same cost but different angle - not dominated
                else:
                    # cost was really higher
                    return True
        # we got through that list without ever breaking, apparently
        return False

    def calc_angle_cost(self, node, angle, new_angle):
        if (node == 0):
            return 0
        else:
            actual_angle = np.abs(angle - new_angle)
            min_angle = np.min([actual_angle, 360 - actual_angle])
            return min_angle  # you could manipulate this further if you wanted:
            # maybe you want to push angles through some function first, exponential weighting or so

    def remove_dominated(self, labels, cost):
        """
        Assumes that (cost, angle) is not dominated in labels!
        """
        removed = []  # can't remove concurrently
        for (label, a) in labels:
            smaller = True
            for i in range(self.d):
                if cost[i] > label[i]:
                    break
            else:  # cost was smaller or equal everywhere
                if np.sum(cost) == np.sum(label):  # they are equal
                    # since we assume that cost is not dominated, we know that both angles need to remain
                    break
                else:
                    removed.append((label, a))

        return [l for l in labels if not l in removed]

    def martins(self, source, target):
        """
        Runs Martin's algorithm from source to target
        """
        s = self.name_to_num[source]
        t = self.name_to_num[target]

        # lists of labels, preliminary and permanent
        labels = [([], []) for _ in range(len(self.nodes))]

        q = []  # not actually a Queue but a heap

        # node at which we are,
        # cost of getting there
        # angle at which we entered it
        heappush(q, Label(s, np.zeros(self.d), 0))

        while q:
            # take candidate out of queue and append to label lists
            (node, cost, angle) = heappop(q).unwrap()
            lv, pv = labels[node]
            try:
                lv.remove((cost, angle))  # wont work for Start
            except:
                pass
            pv.append((cost, angle))
            # this is certainly a non-dominated path to that node
            print(
                "Found path to node ", self.num_to_name[node], " with cost ",
                cost
            )
            labels[node] = (lv, pv)

            # for all nodes to which our node has an edge
            for i, vec in enumerate(self.edge_matrix[node]):
                if vec[0] == 1:

                    add_cost = vec[1:self.d + 1]
                    output_angle = vec[
                        self.d + 1]  # this is the output angle of the edge
                    input_angle = vec[self.d + 2]

                    cost_sum = [a + b for (a, b) in zip(cost, add_cost)]
                    cost_sum[self.d - 1] += self.calc_angle_cost(
                        node, angle, output_angle
                    )

                    # put the follower into label list if it is not dominated
                    # and remove the labels that it dominates
                    Lv, Pv = labels[i]
                    if not self.dominance_check(Lv, cost_sum, output_angle
                                                ) and not self.dominance_check(
                                                    Pv, cost_sum, output_angle
                                                ):
                        Lv = self.remove_dominated(Lv, cost_sum)
                        Lv.append((cost_sum, input_angle))
                        labels[i] = (Lv, Pv)

                        heappush(q, Label(i, cost_sum, input_angle))
        print(labels[t])


if __name__ == "__main__":

    # simple example, route from A to D via C should be
    # more expensive than via B because the angle is sharper

    nodes = [("A", 2, 5), ("B", 3, 3), ("C", 3, 9), ("D", 4, 5), ("E", 6, 1)]

    edges = [
        ("A", "B", np.array([1, 2, 3, 0])), ("A", "C", np.array([1, 2, 3, 0])),
        ("B", "D", np.array([1, 0, 1, 0])), ("C", "D", np.array([1, 0, 1, 0])),
        ("D", "E", np.array([1, 1, 1, 0]))
    ]

    g = Graph(nodes, edges, 4)
    g.martins("A", "E")

# MY NOTES:
# ("C", "D", np.array([1,0,1,0] --> last zero is a placeholder for the angle
# cost! (Added then in line 194)
# dijkstra but update dominance

# could in principle do that directly in bellman ford --> check dominated in
# each iteration

# Take the lexicographically smallest element from the queue
# does it work with the angles? do we check all of them?

# cost_sum is a vector with the accumulated costs at each step