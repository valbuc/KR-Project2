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
import time


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

    def ordering_random(self, originalbn):
        variables = originalbn.get_all_variables()
        random.shuffle(variables)
        return variables

    def ordering_mindegree(self, originalbn) -> List[str]:
        """
        returns ordering of elimination list [order]
        the order is based on first eliminating the nodes with the least amount of neighbors
        """
        bn = copy.deepcopy(originalbn)
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

    def ordering_minfill(self, originalbn) -> List[str]:
        """
        returns ordering of elimination list
        the min-full heuristic creates the order based on first eliminating the variables that add the smallest number of edges
        """
        bn = copy.deepcopy(originalbn)
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
            # TODO also update cpts for variable itself
            # cpt = cp_bn.get_cpt(variable)
            # new_cpt = cp_bn.get_compatible_instantiations_table(e, cpt)
            # print(new_cpt)
            # print(type(new_cpt))
            # new_cpt = new_cpt.loc[new_cpt, "p"] = 1
            # cp_bn.update_cpt(variable, new_cpt)

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
        if len(y) == 0:
            summed_out = factor.agg("sum").to_frame().T
        else:
            summed_out = factor.groupby(y).agg("sum").reset_index()

        # remove variables in z from dataframe
        for variable in variables:
            delete = []
            if variable in summed_out.columns:
                delete.append(variable)
            for var in delete:
                summed_out = summed_out.drop(var, axis=1)

        return summed_out

    def mult(self, factors: list):
        """
        multiplies multiple factors independent of their variable values 
        """
        rowsmult = 0

        variables = []
        for factor in factors:
            for var in factor.columns:
                if var not in variables:
                    variables.append(var)
        variables.remove("p")

        grand = factors[0]
        for factor in factors[1:]:
            intersect = [var for var in grand.columns if var in factor.columns]
            intersect.remove("p")
            if len(intersect) == 0:
                newgrand = pd.DataFrame()
                for row1 in grand.iterrows():
                    for row2 in factor.iterrows():
                        # rename p row of row2
                        row1ser = row1[1].rename({"p": "p_x"})
                        row2ser = row2[1].rename({"p": "p_y"})
                        newrow = row1ser.append(row2ser)
                        newrow = pd.DataFrame(newrow).T
                        newgrand = newgrand.append(newrow)
                grand = newgrand
            else:
                grand = grand.merge(factor, how="outer", on=intersect)
            grand["p"] = grand.apply(
                lambda row: row["p_x"] * row["p_y"], axis=1)
            grand = grand.drop(["p_x", "p_y"], axis=1)
            rowsmult += len(grand)

        return grand, rowsmult

    def create_truth_table(self, num_vars):
        return pd.DataFrame(list(itertools.product([False, True], repeat=num_vars)))

    def get_marginal_distribution(
        self,
        heuristic: str = "random",
        q_vars: list = [],
        e_vars: pd.Series = pd.Series(),
    ):
        """
        heuristic can be 'random', 'mindegree', 'minfill'
        """

        heuristics = {
            "random": self.ordering_random,
            "mindegree": self.ordering_mindegree,
            "minfill": self.ordering_minfill,
        }

        # q_vars = self.bn.get_all_variables()

        N = self.net_prune(q_vars, e_vars)  # prune edges

        # make map vars appear last
        order = heuristics[heuristic](
            N
        )  # elimination order of variables Q # put this as parameter
        for var in q_vars:
            order.remove(var)

        cpts = N.get_all_cpts()

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
            factor, rowsmult = self.mult(factors)

            # sum out variable
            newfactor = self.sum_out(factor, [variable])

            # delete factors from cpts
            for var in delete:
                del cpts[var]

            # add new factor to cpts
            # TODO: can maxxout always return dataframe?
            if type(newfactor) == pd.DataFrame:
                cpts[variable] = newfactor
            else:
                cpts[variable] = newfactor.to_frame().T

        result, rowsmult = self.mult(list(cpts.values()))

        # normalise results
        summ = result["p"].sum()
        result['p'] = result.apply(lambda row: row['p']/summ, axis=1)

        return result

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

        sorting = copy.deepcopy(factor)
        if len(stayvariables) != 0:
            # TODO this should actually only return the row with max p, code for this is at end of MPE function
            sorting = factor.groupby(stayvariables)

        maxx = sorting.max()
        if type(maxx) == pd.Series:
            maxx = maxx.to_frame().T
        maxx = maxx.drop(maxoutvariables, axis=1)

        maxx = maxx.merge(factor, "left", on=["p", *stayvariables])

        maxx = maxx.drop_duplicates()

        return maxx

    def MPE(self, heuristic: str = "random", e_vars: pd.Series = pd.Series()):
        """
        heuristic can be 'random', 'mindegree', 'minfill'

        """

        start = time.time()

        # performance measures
        rows_multiplied = 0
        rows_summed = 0
        rows_maxxed = 0

        heuristics = {
            "random": self.ordering_random,
            "mindegree": self.ordering_mindegree,
            "minfill": self.ordering_minfill,
        }

        q_vars = self.bn.get_all_variables()

        N = self.net_prune(q_vars, e_vars)  # prune edges

        order = heuristics[heuristic](
            N
        )  # elimination order of variables Q # put this as parameter

        cpts = N.get_all_cpts()

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
            factor, rowsmult = self.mult(factors)
            rows_multiplied += rowsmult

            # may out variable
            rows_maxxed += len(factor)
            maxfactor = self.maxx_out(factor, [variable])

            # delete factors from cpts
            for var in delete:
                del cpts[var]

            # add new factor to cpts
            # TODO: can maxxout always return dataframe?
            if type(maxfactor) == pd.DataFrame:
                cpts[variable] = maxfactor
            else:
                cpts[variable] = maxfactor.to_frame().T

        maxx, rowsmult = self.mult(list(cpts.values()))
        m = maxx["p"].max()
        result = maxx.loc[maxx["p"] == m]

        end = time.time()

        elapsed = end - start

        return result, rows_multiplied, rows_summed, rows_maxxed, elapsed

    def MAP(
        self,
        heuristic: str = "random",
        map_vars: list = [],
        e_vars: pd.Series = pd.Series(),
    ):
        """
        heuristic can be 'random', 'mindegree', 'minfill'
        """

        start = time.time()

        # performance measures
        rows_multiplied = 0
        rows_summed = 0
        rows_maxxed = 0

        heuristics = {
            "random": self.ordering_random,
            "mindegree": self.ordering_mindegree,
            "minfill": self.ordering_minfill,
        }

        # q_vars = self.bn.get_all_variables()

        N = self.net_prune(map_vars, e_vars)  # prune edges

        # make map vars appear last
        order = heuristics[heuristic](
            N
        )  # elimination order of variables Q # put this as parameter
        for var in map_vars:
            order.remove(var)
            order.append(var)

        cpts = N.get_all_cpts()

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
            factor, rowsmult = self.mult(factors)
            rows_multiplied += rowsmult

            # if in mapvariables, max out
            if variable in map_vars:
                rows_maxxed += len(factor)
                newfactor = self.maxx_out(factor, [variable])
            else:
                # sum out variable
                rows_summed += len(factor)
                newfactor = self.sum_out(factor, [variable])

            # delete factors from cpts
            for var in delete:
                del cpts[var]

            # add new factor to cpts
            # TODO: can maxxout always return dataframe?
            if type(newfactor) == pd.DataFrame:
                cpts[variable] = newfactor
            else:
                cpts[variable] = newfactor.to_frame().T

        maxx, rowsmult = self.mult(list(cpts.values()))

        # normalize
        summ = maxx["p"].sum()
        maxx['p'] = maxx.apply(lambda row: row['p']/summ, axis=1)

        m = maxx["p"].max()
        result = maxx.loc[maxx["p"] == m]

        end = time.time()

        elapsed = end - start

        return result, rows_multiplied, rows_summed, rows_maxxed, elapsed


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
