from BayesNet import BayesNet
from BNReasoner import BNReasoner
import networkx as nx
import copy
import matplotlib.pyplot as plt

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



bn = copy.deepcopy(dogproblem)

var_neighbor = {}
smallest = 0
order=[]
edges = bn.get_all_edges()
print("the edges were:----", edges)
for variable in bn.get_all_variables():
    print("the currently selected variable is", variable)
    count = 0
    for tuple in edges:
        if variable in tuple:
            count +=1
        var_neighbor[variable]=count
    least = str(min(var_neighbor, key=var_neighbor.get)) #variable with least amount of neighbors
    print("the current var with least amount of neighbors is:------", least)
    ### if this variable appears in one tuple, that means you create no new edges
    print(var_neighbor[variable])
    new_edges_list=[]
    if var_neighbor[variable] != 1:
        for tuple in edges:
            if variable in tuple:
                new_edges_list.append(variable)
                print(new_edges_list)
                



    ### if this variable appears in more than one tuple, you make an edge with the other values in these tuples


    # temp_list=bn.get_all_variables()
    # print(temp_list)
    # temp_list.remove(least)
    # print(temp_list)
    # for var1 in temp_list:
    #     print("this is var1:", var1)
    #     for var2 in temp_list:
    #     # var2=temp_list[-1]
    #         print("this is var2:", var2)
    #         if (var1, var2) in edges:
    #             pass
    #         try:
    #             bn.add_edge((var1, var2))
    #         except: 
    #             pass

edges= bn.get_all_edges()
print("the edges are:----", edges)
bn.draw_structure()
graph = bn.get_structure().to_undirected()

        # for var1 in temp_list[-1]:
        #     print(var1)
        #     bn.add_edge((var, var1))
            
# the edges were:---- [('bowel-problem', 'dog-out'), ('dog-out', 'hear-bark'), ('family-out', 'light-on'), ('family-out', 'dog-out')]

# the edges are:---- [('bowel-problem', 'dog-out'), ('bowel-problem', 'hear-bark'), ('bowel-problem', 'family-out'), ('dog-out', 'hear-bark'), ('family-out', 'dog-out'), ('family-out', 'hear-bark')]



    
