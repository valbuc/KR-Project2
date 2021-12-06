from typing import List, Union
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



    def ordering_mindegree(self) -> List[str]:

        """
        returns ordering of elimination list
        the min-degree heuristic creates the order based on first eliminating the nodes with the least amount of neighbors
        """

        bn = copy.deepcopy(self.bn)
        self.order=[]

        count = 1
        for variable in bn.get_all_variables():
            edges = bn.get_all_edges()
            print("this is iteration:---", count)
            print("the current variable:---", variable)

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                print("this is the ordering:---", self.order)
                return(self.order)
            else:
                pass

            # makes dict which holds variables and amount of edges they have                
            first = Counter(elem[0] for elem in edges)
            second = Counter(elem[1] for elem in edges)
            firstdict = dict(first)
            seconddict = dict(second)
            var_neighbor = {k: firstdict.get(k, 0) + seconddict.get(k, 0) for k in set(firstdict) | set(seconddict)}
            print("variables and amount edges at start of iteration:---", var_neighbor)
                
            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get)) 
            print("the variable with the least edges is:---", least)
            self.order.append(least)
            new_edges_list=[]

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            # if only has one connection, continue
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


            bn.draw_structure()                              
            del var_neighbor[least] # deletes variable from dict
            bn.del_var(least) # deletes variable from bn
            count += 1
            print("updated dict:---", var_neighbor)


    def ordering_minfull(self) -> List[str]:

        """
        WORK IN PROGRESS
        returns ordering of elimination list
        the min-full heuristic creates the order based on first eliminating the variables that add the smallest number of edges
        """

        bn = copy.deepcopy(self.bn)
        self.order=[]

        count = 1
        for variable in bn.get_all_variables():
            edges = bn.get_all_edges()
            print("this is iteration:---", count)
            print("the current variable:---", variable)

            # when at final variable, returns list
            if len(edges) == 0:
                self.order.append(variable)
                print("this is the ordering:---", self.order)
                return(self.order)
            else:
                pass

            # makes dict which holds variables and amount of edges they have                
            first = Counter(elem[0] for elem in edges)
            second = Counter(elem[1] for elem in edges)
            firstdict = dict(first)
            seconddict = dict(second)
            var_neighbor = {k: firstdict.get(k, 0) + seconddict.get(k, 0) for k in set(firstdict) | set(seconddict)}
            print("variables and amount edges at start of iteration:---", var_neighbor)
                
            # selects variable with least amount of edges/neighbor
            least = str(min(var_neighbor, key=var_neighbor.get)) 
            print("the variable with the least edges is:---", least)
            self.order.append(least)
            new_edges_list=[]

            # if variable has non-adjacdent neighbors (more than one edge connection), add edges between them
            # if only has one connection, continue

            # stairway to literal hell 
            edges_to_add=0
            connect_dict={}
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
                                                edges_to_add += 1
                                                connect_dict.append[var1]=edges_to_add
                                            except: 
                                                pass

                least = str(min(edges_to_add, key=edges_to_add.get))
                del edges_to_add[least]
            else:
                del var_neighbor[least]

            bn.draw_structure()                              
            bn.del_var(least) # deletes variable from bn
            count += 1
            print("updated dict:---", var_neighbor)