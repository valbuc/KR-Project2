from typing import List, Union
from BayesNet import BayesNet
import networkx as nx
import copy
import pandas as pd
from collections import Counter
import itertools
import numpy as np


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

    def create_truth_table(self, num_vars):
        return pd.DataFrame(list(itertools.product([False, True], repeat= num_vars)))

    def multiply(self, factor1: pd.DataFrame, factor2: pd.DataFrame):
        """
        takes 2 cpt(factors)
        returns cpt(factor1) multiplied by cpt(factor2)
        """

        vars1 = list(factor1.columns)
        vars1.remove('p')
        vars2 = list(factor2.columns)
        vars2.remove('p')

        multiplied = copy.deepcopy(factor1)
        new_in_vars2 = set(vars2).difference(set(vars1))

        # if there is a variable in factor2 that is not in factor1 then factor1 has to be extended by those variable(s)
        if len(new_in_vars2) > 0:
            for var in new_in_vars2:
                if True in factor2[var] and False in factor2[var]:
                    multiplied_copy = copy.deepcopy(multiplied)
                    multiplied.insert(len(factor1.columns) - 1, var, True)
                    multiplied = pd.concat([multiplied, multiplied_copy])
                    multiplied = multiplied.fillna(False)
                else:
                    if True in factor2[var]:
                        multiplied.insert(len(factor1.columns)-1, var, True)
                    if False in factor2[var]:
                        multiplied.insert(len(factor1.columns)-1, var, False)


            cols = list(multiplied.columns)
            multiplied = multiplied.sort_values(by=cols, ascending=False)
            multiplied = multiplied.reset_index(drop=True)

        # compare the columns included in factor2 (vars2) row by row between factor1 and factor2
        # if they are the same then the 2 probabilities can be multiplied
        for r1, row1 in multiplied.iterrows():
            for r2, row2 in factor2.iterrows():
                if list(row1[list(vars2)]) == list(row2[list(vars2)]):
                    multiplied.at[r1, 'p'] *= row2['p']
                    break

        return multiplied


    def get_marignal(self, q_vars: list, e_vars: pd.DataFrame):
        cpts = self.bn.get_all_cpts()

        #make cpts consistent with evidence (delete inconsistent rows)
        for key in cpts:
            relevant_evidence = []
            for var in e_vars:
                if var in cpts[key]:
                    relevant_evidence.append(var)
            to_delete = []
            if relevant_evidence != []:
                for r, row in cpts[key].iterrows():
                    if list(row[relevant_evidence]) != list(e_vars[relevant_evidence].iloc[0]):
                        to_delete.append(r)
                cpts[key] = cpts[key].drop(to_delete, axis=0)

        for key1 in cpts:
            if key1 not in q_vars:
                for key2 in cpts:
                    if key2 != key1 and key1 in cpts[key2]:
                        cpts[key2] = reasoner.multiply(cpts[key2], cpts[key1])
                        cpts[key2] = reasoner.sum_out(cpts[key2], [key1])

        # delete everything that is not in q_vars
        to_delete = [key for key in cpts if key not in q_vars]
        for var in to_delete:
            cpts.pop(var)

        # normalise results
        for key in cpts:
            cpts[key] = cpts[key]['p'] / cpts[key]['p'].sum()
        return cpts



if __name__ == '__main__':
    dogproblem = BayesNet()
    dogproblem.load_from_bifxml('testing/dog_problem.BIFXML')
    #dogproblem.draw_structure()
    reasoner = BNReasoner(dogproblem)
    dog = dogproblem.get_cpt('dog-out')
    bark = dogproblem.get_cpt('hear-bark')
    multiplied = reasoner.multiply(bark, dog)


    evidence = pd.DataFrame([{'bowel-problem': False, 'family-out': False}])
    query_vars = ['dog-out', 'light-on']

    reasoner.get_marignal(query_vars, evidence)