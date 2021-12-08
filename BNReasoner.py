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
            for var in bn.get_all_variables():
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num

            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get))
            self.order.append(least)

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            if var_neighbor[least] != 1: 
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

    def ordering_minfull(self) -> List[str]:
        """
        returns ordering of elimination list
        the min-full heuristic creates the order based on first eliminating the variables that add the smallest number of edges
        """
        bn = copy.deepcopy(self.bn)
        self.order = []

        count = 1
        for variable in bn.get_all_variables():
            bn.draw_structure()
            edges = bn.get_all_edges()
            var_neighbor = {}
            G = bn.get_structure().to_undirected()

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                return self.order
            else:
                pass

            # makes dict which holds variables and amount of edges they have
            for var in bn.get_all_variables():
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num
               
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
                
                all = [(leasts_neighbors[i],leasts_neighbors[j]) for i in range(len(leasts_neighbors)) for j in range(i+1, len(leasts_neighbors))]
                for tuple in all:
                    if tuple in edges:
                        pass
                    else:
                        try:
                            bn_copy.add_edge(tuple) 
                            edges_to_add += 1  
                            edges_amount[least] = edges_to_add 
                        except: 
                            pass
                least = str(min(edges_amount, key=edges_amount.get))

                # the actual creation of the new edges
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
            bn.del_var(least)  
            count += 1

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
