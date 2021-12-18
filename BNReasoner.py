from typing import List, Union
from BayesNet import BayesNet
import networkx as nx
import copy
import pandas as pd
from collections import Counter
import itertools
import numpy as np
import matplotlib.pyplot as plt
import random


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

    def ordering_random(self):
        variables = self.bn.get_all_variables()
        random.shuffle(variables)
        return variables

    def ordering_mindegree(self) -> List[str]:
        """
        returns ordering of elimination list [order]
        the order is based on first eliminating the nodes with the least amount of neighbors
        """
        bn = copy.deepcopy(self.bn)
        G = bn.get_interaction_graph()
        self.order = []

        for i in range(len(bn.get_all_variables())):
            nodes = list(G.nodes)
            # nx.draw(G, with_labels=True)
            plt.show()

            # makes dict which holds variables and amount of edges they have
            var_neighbor = {}
            for var in nodes:
                num = len(list(nx.neighbors(G, n=var)))
                var_neighbor[var] = num

            # selects variable with least amount of edges/neighbors, then appends to list
            least = str(min(var_neighbor, key=var_neighbor.get))
            self.order.append(least)

            # removes said variable from both BN and interaction graph
            bn.del_var(least)
            G.remove_node(least)

        return self.order

    def ordering_minfill(self) -> List[str]:
        """
        returns ordering of elimination list
        the min-full heuristic creates the order based on first eliminating the variables that add the smallest number of edges
        """
        bn = copy.deepcopy(self.bn)
        G = bn.get_interaction_graph()
        self.order = []

        for i in range(len(bn.get_all_variables())):
            nodes = list(G.nodes)
            # nx.draw(G, with_labels=True)
            plt.show()

            edges_to_add = {}
            real_edges = len(list(G.edges))

            # creates a fake BN and G to simulate how many edges would have to be added if said var was deleted
            for var in nodes:
                bn_copy = copy.deepcopy(bn)
                G_copy = copy.deepcopy(G)
                bn_copy.del_var(var)
                G_copy.remove_node(var)

                # gets the difference between how many edges there were before, and how many edges there would be after deletion
                fake_edges = len(list(G_copy.edges))
                diff = real_edges - fake_edges

                # adds this difference to dictionary
                edges_to_add[var] = diff

            # selects the var which has the lowest difference, aka the variable that's deletion would add least edges
            least = str(min(edges_to_add, key=edges_to_add.get))
            self.order.append(least)

            # actually deletes this variable
            bn.del_var(least)
            G.remove_node(least)

        return self.order

    def net_prune(self, q: list, e: pd.Series):
        """
        Network Pruning is done in two parts: Node Pruning and Edge Pruning.
        """
        new_e = []
        for items in e.iteritems():
            new_e.append(items[0])

        qe = q + new_e
        cp_bn = copy.deepcopy(self.bn)
        # cp_bn.draw_structure()

        # Node pruning
        while True:
            sett = cp_bn.get_all_variables()

            for variable, _ in cp_bn.get_all_edges():
                if variable in sett:
                    sett.remove(variable)

            for var in qe:
                if var in sett:
                    sett.remove(var)

            if len(sett) == 0:
                break

            for item in sett:
                cp_bn.del_var(item)

        # cp_bn.draw_structure()

        # Edge pruning
        for variable in new_e:
            for child in cp_bn.get_children(variable):
                cp_bn.del_edge((variable, child))

                cpt = cp_bn.get_cpt(child)
                new_cpt = cp_bn.get_compatible_instantiations_table(e, cpt)
                for ev in e.iteritems():
                    if ev[0] in new_cpt.columns:
                        new_cpt = new_cpt.drop(ev[0], axis=1)
                cp_bn.update_cpt(child, new_cpt)

        # cp_bn.draw_structure()

        return cp_bn

    def sum_out(self, factor: pd.DataFrame, variables: list):
        """
        takes a cpt(factor) and a set of variables
        returns a cpt with the goven variables summed out
        """

        # getting all variables in the factor
        x = list(factor.columns)
        x.remove("p")

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

    def mult(self, factors: list):
        """
        multiplies multiple factors independent of their variable values 
        """

        variables = []
        for factor in factors:
            for var in factor.columns:
                if var not in variables:
                    variables.append(var)
        variables.remove("p")

        intersect = set(factors[0].columns)
        for factor in factors[1:]:
            intersect.intersection_update(factor.columns)
        intersect = list(intersect)
        intersect.remove("p")

        # TODO make it possible to multiply factors without overlap
        if len(intersect) == 0:
            grand = factors[0]
            for factor in factors[1:]:
                pass

        grand = factors[0]
        for factor in factors[1:]:
            grand = grand.merge(factor, how="outer", on=intersect)
            grand["p"] = grand.apply(lambda row: row["p_x"] * row["p_y"], axis=1)
            grand = grand.drop(["p_x", "p_y"], axis=1)

        return grand

    def create_truth_table(self, num_vars):
        return pd.DataFrame(list(itertools.product([False, True], repeat=num_vars)))

    def multiply(self, factor1: pd.DataFrame, factor2: pd.DataFrame):
        """
        takes 2 cpt(factors)
        returns cpt(factor1) multiplied by cpt(factor2)
        """

        vars1 = list(factor1.columns)
        vars1.remove("p")
        vars2 = list(factor2.columns)
        vars2.remove("p")

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
                        multiplied.insert(len(factor1.columns) - 1, var, True)
                    if False in factor2[var]:
                        multiplied.insert(len(factor1.columns) - 1, var, False)

            cols = list(multiplied.columns)
            multiplied = multiplied.sort_values(by=cols, ascending=False)
            multiplied = multiplied.reset_index(drop=True)

        # compare the columns included in factor2 (vars2) row by row between factor1 and factor2
        # if they are the same then the 2 probabilities can be multiplied
        for r1, row1 in multiplied.iterrows():
            for r2, row2 in factor2.iterrows():
                if list(row1[list(vars2)]) == list(row2[list(vars2)]):
                    multiplied.at[r1, "p"] *= row2["p"]
                    break

        return multiplied

    def multiply_multi(self, *args):
        """
        multiplies >2 factors 
        """

        # print(list(args))

        multiplied = list(args)[0]

        for factor in list(args)[1:]:
            multiplied = self.multiply(multiplied, factor)

        return multiplied

    def get_marginal(self, q_vars: list, e_vars: pd.DataFrame):
        cpts = self.bn.get_all_cpts()

        # make cpts consistent with evidence (delete inconsistent rows)
        for key in cpts:
            relevant_evidence = []
            for var in e_vars:
                if var in cpts[key]:
                    relevant_evidence.append(var)
            to_delete = []
            if relevant_evidence != []:
                for r, row in cpts[key].iterrows():
                    if list(row[relevant_evidence]) != list(
                        e_vars[relevant_evidence].iloc[0]
                    ):
                        to_delete.append(r)
                cpts[key] = cpts[key].drop(to_delete, axis=0)

        for key1 in cpts:
            if key1 not in q_vars:
                for key2 in cpts:
                    if key2 != key1 and key1 in cpts[key2]:
                        cpts[key2] = self.multiply(cpts[key2], cpts[key1])
                        cpts[key2] = self.sum_out(cpts[key2], [key1])

        # delete everything that is not in q_vars
        to_delete = [key for key in cpts if key not in q_vars]
        for var in to_delete:
            cpts.pop(var)

        # normalise results
        for key in cpts:
            cpts[key] = cpts[key]["p"] / cpts[key]["p"].sum()
            cpts[key] = cpts[key].to_frame()
            cpts[key][key] = [False, True]

        factors = list(cpts.values())

        marginalpt = self.multiply_multi(*factors)

        return marginalpt

    def maxx_out(self, factor: pd.DataFrame, maxoutvariables: list):
        """
        takes a cpt(factor) and a set of variables
        returns a cpt with the given variables maxxed out
        """

        # getting all variables in the factor
        allvariables = list(factor.columns)
        allvariables.remove("p")

        # get a list of variables which should remain
        stayvariables = [
            variable for variable in allvariables if variable not in maxoutvariables
        ]

        if len(stayvariables) == 0:
            # TODO this should actually only return the row with max p, code for this is at end of MPE function
            return factor.max()
        print(factor)
        sorting = factor.groupby(stayvariables)
        maxx = sorting.max()
        print(maxx)
        maxx = maxx.drop(maxoutvariables, axis=1)

        maxx = maxx.merge(factor, "left", on=["p", *stayvariables])

        maxx = maxx.drop_duplicates()
        print(maxx)

        return maxx

    def MPE(self, heuristic: str = "random", e_vars: pd.Series = pd.Series()):
        """
        heuristic can be 'random', 'mindegree', 'minfill'
        
        """

        heuristics = {
            "random": self.ordering_random,
            "mindegree": self.ordering_mindegree,
            "minfill": self.ordering_minfill,
        }

        q_vars = self.bn.get_all_variables()

        N = self.net_prune(q_vars, e_vars)  # prune edges

        order = heuristics[
            heuristic
        ]()  # elimination order of variables Q # put this as parameter

        cpts = N.get_all_cpts()

        print(order)

        # loop over variables in order given
        for variable in order:
            # get factors which contain variable
            factors = []
            delete = []
            for key, value in cpts.items():
                if variable in value.columns:
                    factors.append(value)
                    delete.append(key)
            if len(factors) == 0:
                continue

            # multiply factors
            print(factors)
            factor = self.mult(factors)

            print(factor)

            # may out variable
            maxfactor = self.maxx_out(factor, [variable])

            # delete factors from cpts
            for var in delete:
                del cpts[var]

            # add new factor to cpts
            # TODO: can maxxout always return dataframe?
            print(maxfactor)
            if type(maxfactor) == pd.DataFrame:
                cpts[variable] = maxfactor
            else:
                cpts[variable] = maxfactor.to_frame().T

        for factor in cpts.values():
            print(factor)
            print(type(factor))

        maxx = self.mult(list(cpts.values()))
        m = maxx["p"].max()
        print(m)
        result = maxx.loc[maxx["p"] == m]

        return result


if __name__ == "__main__":
    dogproblem = BayesNet()
    dogproblem.load_from_bifxml("testing/dog_problem.BIFXML")
    # dogproblem.draw_structure()
    reasoner = BNReasoner(dogproblem)
    dog = dogproblem.get_cpt("dog-out")
    bark = dogproblem.get_cpt("hear-bark")
    multiplied = reasoner.multiply(bark, dog)

    evidence = pd.DataFrame([{"bowel-problem": False, "family-out": False}])
    query_vars = ["dog-out", "light-on"]

    reasoner.get_marginal(query_vars, evidence)
