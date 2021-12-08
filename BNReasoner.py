from typing import List, Union
from BayesNet import BayesNet
import networkx as nx
import copy
import pandas as pd
from collections import Counter


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

    # TODO: This is where your methods should go

    def d_separation(self, x: list, y: list, z: list) -> bool:
        """
        given stes of variables x, y, and z, returns wether x is independent
        of y given z
        """
        xyz = x + y + z
        cpbn = copy.deepcopy(self.bn)
        # bn.draw_structure()
        # first save all items to be delted in list and then iterate over that list to delete
        while True:
            delete = []
            continueiter = False
            for variable in cpbn.get_all_variables():
                # children = [1]
                # try:
                children = cpbn.get_children(variable)
                # except:
                #    pass
                if len(children) == 0:
                    if variable not in xyz:
                        delete.append(variable)
                        # try:
                        #   bn.del_var(variable)
                        # except:
                        #    pass
            if len(delete) == 0:
                break

            for variable in delete:
                cpbn.del_var(variable)
                delete_edge = []
                for edge in cpbn.get_all_edges():
                    if variable in edge:
                        delete_edge.append(edge)
                for edge in delete_edge:
                    cpbn.del_edge(edge)

        for variable in z:
            for edge in cpbn.get_all_edges():
                if edge[0] == variable:
                    cpbn.del_edge(edge)

        # bn.draw_structure()
        graph = cpbn.get_structure().to_undirected()
        # graph = bn.get_interaction_graph()

        for X in x:
            for Y in y:
                haspath = nx.has_path(graph, X, Y)
                if haspath:
                    return False

        return True

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
            var_neighbor = {}

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                return self.order
            else:
                pass

            # makes dict which holds variables and amount of edges they have
            for i in bn.get_all_variables():
                num = len(list(nx.neighbors(G, n=i)))
                var_neighbor[i] = num

            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            self.order.append(least)

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            # if only has one connection, continue
            if var_neighbor[least] != 1: 
                leasts_neighbors = list(nx.neighbors(G, n=i))
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

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                print("this is the ordering:---", self.order)
                return self.order
            else:
                pass

            # makes dict which holds variables and amount of edges they have
            first = Counter(elem[0] for elem in edges)
            second = Counter(elem[1] for elem in edges)
            firstdict = dict(first)
            seconddict = dict(second)
            var_neighbor = {
                k: firstdict.get(k, 0) + seconddict.get(k, 0)
                for k in set(firstdict) | set(seconddict)
            }
            print("variables and amount edges at start of iteration:---", var_neighbor)

            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            print("the variable with the least edges is:---", least)
            self.order.append(least)
            new_edges_list = []

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            # if only has one connection, continue

            # stairway to literal hell
            edges_to_add = 0
            connect_dict = {}
            if var_neighbor[least] != 1:
                for tuple in edges:
                    if least in tuple:
                        for i in tuple:
                            if i != least:
                                new_edges_list.append(i)
                                for var1 in new_edges_list:
                                    for var2 in new_edges_list:
                                        var2 = new_edges_list[-1]
                                        if (var1, var2) in edges:
                                            pass
                                            try:
                                                bn.add_edge((var1, var2))
                                                edges_to_add += 1
                                                connect_dict.append[var1] = edges_to_add
                                            except:
                                                pass

                least = str(min(edges_to_add, key=edges_to_add.get))
                del edges_to_add[least]
            else:
                del var_neighbor[least]

            bn.del_var(least)  # deletes variable from bn
            count += 1
            print("updated dict:---", var_neighbor)

    def sum_out(self, factor: pd.DataFrame, variables: list):
        """
        takes a cpt(factor) and a set of variables
        returns a cpt with the goven variables summed out
        """

        # getting all variables in the factpr
        x = list(factor.columns)
        x.pop()

        # get a list of variables which should remain
        y = [X for X in x if X not in variables]

        # sum out variable
        summed_out = factor.groupby(y).agg("sum").reset_index()

        # remove variables in z from dataframe
        for variable in variables:
            delete = []
            if variable in summed_out.columns:
                delete.append(variable)
            for var in delete:
                summed_out = summed_out.drop(var, 1)

        return summed_out
