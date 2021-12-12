from typing import List, Union
from BayesNet import BayesNet
import networkx as nx
import copy
from collections import Counter
import itertools 


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

        count = 1
        for variable in bn.get_all_variables():
            G = bn.get_structure().to_undirected()
            bn.draw_structure()
            edges = bn.get_all_edges()

            print("this is iteration:---", count)
            print("the current variable:---", variable)
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
            for var in bn.get_all_variables():
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num
            print("var and numbers are", var_neighbor)

            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            print("the variable with the least edges is:---", least)
            self.order.append(least)

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            # if only has one connection, continue
            # going to work on making this more efficient
            if var_neighbor[least] != 1: #if more than one neighbor
                leasts_neighbors = list(nx.neighbors(G, n=least))
                print(leasts_neighbors)
                all = [(leasts_neighbors[i],leasts_neighbors[j]) for i in range(len(leasts_neighbors)) for j in range(i+1, len(leasts_neighbors))]
                for tuple in all:
                    if tuple in edges:
                        pass
                    else:
                        try:
                            bn.add_edge(tuple)
                        except: 
                            pass
            del var_neighbor[least]  # deletes variable from dict
            bn.del_var(least)  # deletes variable from bn
            count += 1
            print("updated dict2:---", var_neighbor)

                # for tuple in edges:
                #     if least in tuple:
                #         for i in tuple:
                #             if i != least:
                #                 new_edges_list.append(i)
                #                 for var1 in new_edges_list:
                #                     for var2 in new_edges_list:
                #                         var2 = new_edges_list[-1]
                #                         if (var1, var2) in edges:
                #                             pass
                #                             try:
                #                                 bn.add_edge((var1, var2))
                #                             except:
                #                                 pass

    def ordering_minfull(self) -> List[str]:

        """
        WORK IN PROGRESS
        returns ordering of elimination list
        the min-full heuristic creates the order based on first eliminating the variables that add the smallest number of edges
        """
        bn = copy.deepcopy(self.bn)
        self.order = []

        count = 1
        for variable in bn.get_all_variables():
            bn.draw_structure()
            edges = bn.get_all_edges()
            print("this is iteration:---", count)
            print("the current variable:---", variable)
            var_neighbor = {}
            G = bn.get_structure().to_undirected()

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                print("this is the ordering:---", self.order)
                return self.order
            else:
                pass

            # makes dict which holds variables and amount of edges they have
            for var in bn.get_all_variables():
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num
               
            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            print("the variable with the least edges is:---", least)
            self.order.append(least)
            new_edges_list = []

            edges_to_add = 0
            edges_amount = {}

            bn_copy = copy.deepcopy(bn)
            G_copy = bn_copy.get_structure().to_undirected()
            if var_neighbor[least] != 1: #if more than one neighbor
                leasts_neighbors = list(nx.neighbors(G_copy, n=least))
                print(leasts_neighbors)
                all = [(leasts_neighbors[i],leasts_neighbors[j]) for i in range(len(leasts_neighbors)) for j in range(i+1, len(leasts_neighbors))]
                for tuple in all:
                    if tuple in edges:
                        pass
                    else:
                        try:
                            bn_copy.add_edge(tuple) #adds edge to copy of bn
                            edges_to_add += 1  
                            edges_amount[least] = edges_to_add #if edge added, add to count of hypothetical edges
                        except: 
                            pass
                least = str(min(edges_amount, key=edges_amount.get))


                leasts_neighbors = list(nx.neighbors(G, n=least))
                print(leasts_neighbors)
                all = [(leasts_neighbors[i],leasts_neighbors[j]) for i in range(len(leasts_neighbors)) for j in range(i+1, len(leasts_neighbors))]
                for tuple in all:
                    if tuple in edges:
                        pass
                    else:
                        try:
                            bn.add_edge(tuple)
                        except: 
                            pass

            del var_neighbor[least]
            bn.del_var(least)  # deletes variable from bn
            count += 1
            # print("updated dict:---", var_neighbor)


            # if var_neighbor[least] != 1:
            #     for tuple in edges:
            #         if least in tuple:
            #             for i in tuple:
            #                 if i != least:
            #                     new_edges_list.append(i)
            #                     for var1 in new_edges_list:
            #                         for var2 in new_edges_list:
            #                             var2 = new_edges_list[-1]
            #                             if (var1, var2) in edges:
            #                                 pass
            #                                 try:
            #                                     bn.add_edge((var1, var2))
            #                                     edges_to_add += 1
            #                                     connect_dict.append[var1] = edges_to_add
            #                                 except:
            #                                     pass

            #     least = str(min(edges_to_add, key=edges_to_add.get))
            #     del edges_to_add[least]
            # else:
            #     del var_neighbor[least]

            # bn.del_var(least)  # deletes variable from bn
            # count += 1
            # print("updated dict:---", var_neighbor)