from typing import Union
from BayesNet import BayesNet
import networkx as nx
import copy
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
        bn = copy.deepcopy(self.bn)
        bn.draw_structure()
        continueiter = True
        while continueiter == True:
            continueiter = False
            for variable in self.bn.get_all_variables():
                children = [1]
                try:
                    children = bn.get_children(variable)
                except:
                    pass
                if len(children) == 0:
                    if variable not in xyz:
                        try:
                            bn.del_var(variable)
                        except:
                            pass
                        for edge in bn.get_all_edges():
                            if variable in edge:
                                bn.del_edge(edge)
                        continueiter = True
            for variable in z:
                for edge in bn.get_all_edges():
                    if edge[0] == variable:
                        bn.del_edge(edge)

        bn.draw_structure()
        graph = bn.get_structure().to_undirected()
        # graph = bn.get_interaction_graph()

        for X in x:
            for Y in y:
                haspath = nx.has_path(graph, X, Y)
                if haspath:
                    return False

        return True



    def ordering_mindegree(self):

        bn = copy.deepcopy(self.bn)
        var_neighbor = {}
        order=[]
        edges = bn.get_all_edges()

        for variable in bn.get_all_variables():

            # when final variable, returns list
            if len(edges) == 0:
                order.append(variable)
                return(self.order)
            else:
                pass

            # makes dict which holds variables and amount of edges they have                
            first = Counter(elem[0] for elem in edges)
            second = Counter(elem[1] for elem in edges)
            firstdict = dict(first)
            seconddict = dict(second)
            var_neighbor = {k: firstdict.get(k, 0) + seconddict.get(k, 0) for k in set(firstdict) | set(seconddict)}
                
            # selects variable with least amount of edges/neighbors
            least = str(min(var_neighbor, key=var_neighbor.get)) 
            order.append(least)
            new_edges_list=[]

            # if variable has non-adjacdent neighbors, add edges between them
            if var_neighbor[least] != 1:
                for tuple in edges:
                    if least in tuple:
                        for i in tuple:
                            if i != least:
                                new_edges_list.append(i)                            
                                for var1 in new_edges_list:                               
                                    for var2 in new_edges_list:
                                        var2=new_edges_list[-1]                                      
                                        if (var1, var2) in edges:
                                            pass
                                            try:
                                                bn.add_edge((var1, var2))
                                            except: 
                                                pass
                    
            del var_neighbor[least] # deletes variable from dict
            bn.del_var(least) # deletes variablefrom bn