from typing import Union
from BayesNet import BayesNet
import networkx as nx
import copy


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
        bn = copy.deepcopy(self.bn)
        bn.draw_structure()
        continueiter = True
        while continueiter == True:
            continueiter = False
            for variable in self.bn.get_all_variables():
                print(variable)
                children = [1]
                try:
                    children = bn.get_children(variable)
                except:
                    pass
                print(children)
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
                print(variable)
                for edge in bn.get_all_edges():
                    if edge[0] == variable:
                        bn.del_edge(edge)

        bn.draw_structure()

        connected = []
        for X in x:
            for Y in y:
                connected.append(nx.has_path(bn.get_interaction_graph(),X,Y))

        return sum(connected) == 0
