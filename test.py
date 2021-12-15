from typing import List, Union

from networkx.classes import graph
from BayesNet import BayesNet
import networkx as nx
import copy
from collections import Counter
import itertools 
import pandas as pd
import matplotlib.pyplot as plt


    #TD: If at some point you get a "omp Error #15," try running this code. It worked for me, but apparently it is not the best solution as it can cause problems, so use with "caution" I suppose 

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format
        or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net


    def ordering_mindegree(self) -> List[str]:

        """
        returns ordering of elimination list
        the min-degree heuristic creates the order based on first eliminating the nodes with the least amount of neighbors
        """
        bn = copy.deepcopy(self.bn)
        self.order = []

        loop = bn.get_all_variables()
        count = 1
        for variable in loop:
            print("this is iteration:---", count)
            G = bn.get_interaction_graph()
            nx.draw(G, with_labels=True)
            print(G)
            plt.show()
            nodes = list(G.nodes)
            edges = list(G.edges)
            print("the nodes are:", nodes)
            print("the edges are:", edges)
            var_neighbor = {}

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                print("this is the ordering:---", self.order)
                return self.order
            else:
                pass

            # makes dict which holds variables and amount of edges they have
            # for var in bn.get_all_variables():
            for var in nodes:
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num
            print("var and numbers are", var_neighbor)

            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            print("the variable with the least edges is:---", least)
            self.order.append(least)

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            # if only has one connection, continue
            # print(var_neighbor[least])
            # if var_neighbor[least] > 1:
            #     leasts_neighbors = list(nx.neighbors(G, n=least))
            #     print(least, "s neighbors are", leasts_neighbors)
            #     all = [(leasts_neighbors[i],leasts_neighbors[j]) for i in range(len(leasts_neighbors)) for j in range(i+1, len(leasts_neighbors))]
            #     print(all)
            #     for tuple in all:
            #         print(tuple)
            #         if tuple in edges:
            #             pass
            #         else:
            #             G.add_edge(tuple)
            print("test", count)
            bn.del_var(least)
            G.remove_node(least)
            count += 1


    def ordering_minfull(self) -> List[str]:
        """
        returns ordering of elimination list
        the min-full heuristic creates the order based on first eliminating the variables that add the smallest number of edges
        """
        bn = copy.deepcopy(self.bn)
        self.order = []

        count = 1
        for variable in bn.get_all_variables():
            print("this is iteration:---", count)
            G = bn.get_interaction_graph()
            nx.draw(G, with_labels=True)
            print(G)
            plt.show()
            nodes = list(G.nodes)
            edges = list(G.edges)
            print("the nodes are:", nodes)
            print("the edges are:", edges)
            var_neighbor = {}
            # bn.draw_structure()
            # edges = bn.get_all_edges()
            # var_neighbor = {}
            # G = bn.get_structure().to_undirected()

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                print("this is the ordering:---", self.order)
                return self.order
            else:
                pass

            # makes dict which holds variables and amount of edges they have
            for var in nodes:
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num
            print("var and numbers are", var_neighbor)


            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            self.order.append(least)

            # if has more than one neighbor, sees how many new edges would have to be made if it was deleted.
            if var_neighbor[least] != 1:
                edges_to_add = 0
                edges_amount = {}
                bn_copy = copy.deepcopy(bn)
                G_copy = bn_copy.get_structure().to_undirected()
                leasts_neighbors = list(nx.neighbors(G_copy, n=least))

                all = [
                    (leasts_neighbors[i], leasts_neighbors[j])
                    for i in range(len(leasts_neighbors))
                    for j in range(i + 1, len(leasts_neighbors))
                ]
                for tuple in all:
                    if tuple in edges:
                        pass
                    else:
                        try:
                            bn_copy.add_edge(tuple)
                            edges_to_add += 1
                            edges_amount[least] = edges_to_add
                            least = str(min(edges_amount, key=edges_amount.get))
                        except:
                            pass
                
                # the actual creation of the new edges
                leasts_neighbors = list(nx.neighbors(G, n=least))
                print(leasts_neighbors)
                all = [
                    (leasts_neighbors[i], leasts_neighbors[j])
                    for i in range(len(leasts_neighbors))
                    for j in range(i + 1, len(leasts_neighbors))
                ]
                for tuple in all:
                    if tuple in edges:
                        pass
                    else:
                        try:
                            bn.add_edge(tuple)
                        except:
                            pass
                        
            del var_neighbor[least]
            bn.del_var(least)
            count += 1


# dogproblem = BayesNet()
# dogproblem.load_from_bifxml('testing/dog_problem.BIFXML')
# reasoner = BNReasoner(dogproblem)

lecture = BayesNet()
lecture.load_from_bifxml('testing/lecture_example.BIFXML')

reasoner = BNReasoner(lecture)




reasoner.ordering_mindegree()