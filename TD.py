from BayesNet import BayesNet
from BNReasoner import BNReasoner
import networkx as nx
import copy
import matplotlib.pyplot as plt
from collections import Counter

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


dogproblem = BayesNet()
dogproblem.load_from_bifxml('testing/dog_problem.BIFXML')
# dogproblem.draw_structure()
dogproblem.get_all_variables()
s= dogproblem.get_all_cpts()

dogproblem.get_children('light-on')
dogproblem.get_all_edges()

#-------------Ordering-----------------------------------------------------------------------------------------------------------------
# Given a set of variables X in the Bayesian network, compute a good ordering of elimination of X based on the min-degree heuristics and the min-fill heuristics.
# Min-degree Ordering

#interaction graphs show the interactions between CPTs of a BN
#nodes --> the variables that appear in factors f1...fn
#edges --> connect the variables which appaer in the same factor


def ordering_mindegree(self):

    bn = copy.deepcopy(self.bn)

    var_neighbor = {}
    order=[]
    edges = bn.get_all_edges()
    print("the edges were:----", edges)

    edges = bn.get_all_edges()
    bn.draw_structure()

    interations = 1
    for variable in bn.get_all_variables():
        print("the currently selected variable is:----", variable)

        # counts the amount of edges said variable has
        count = 0
        # for variable in bn.get_all_variables():
        ok = Counter(elem[0] for elem in edges)
        bruh = Counter(elem[1] for elem in edges)
        okdict = dict(ok)
        bruhdict = dict(bruh)
        var_neighbor = {k: okdict.get(k, 0) + bruhdict.get(k, 0) for k in set(okdict) | set(bruhdict)}
        # print("this is the dict:---", var_neighbor)
            
        
        # selects variable with least amount of edges/neighbors
        least = str(min(var_neighbor, key=var_neighbor.get)) 
        order.append(least)
        print("the current var with least amount of neighbors is:------", least)
        new_edges_list=[]

        # if variable has non-adjacdent neighbors, add edges between them
        if var_neighbor[least] != 1:
            for tuple in edges:
                if least in tuple:
                    for i in tuple:
                        if i != least:
                            new_edges_list.append(i)
                            print(new_edges_list)
                            for var1 in new_edges_list:
                                print("this is var1:", var1)
                                for var2 in new_edges_list:
                                    var2=new_edges_list[-1]
                                    print("this is var2:", var2)
                                    if (var1, var2) in edges:
                                        pass
                                        try:
                                            bn.add_edge((var1, var2))
                                        except: 
                                            pass

        del var_neighbor[least] #deletes from dict
        print("updated dict:----", var_neighbor)
        bn.del_var(least) # deletes from bn
        edges= bn.get_all_edges()
        bn.draw_structure()
        graph = bn.get_structure().to_undirected()
        print("the order is:", order)
        print("the edges are:----", edges)
        




    edges= bn.get_all_edges()
    print("the edges are:----", edges)
    bn.draw_structure()
    graph = bn.get_structure().to_undirected()
    print(least)
    if len(edges) == 0:
        return




    
