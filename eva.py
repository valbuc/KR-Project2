from typing import Union
from BayesNet import BayesNet
from BNReasoner import BNReasoner
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


    def prune(self, q: list, e:list) -> bool:
        '''
        method that:  
            deletes every leaf node of all variables.
            deletes all edges outgoing from nodes in E.

        Given a set of query variables Q and evidence E, 
        node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated
        '''
        qe = q + e
        cp_bn = copy.deepcopy(self.bn)
        # cp_bn.draw_structure()
            
        # gets the variables and put them in a set   
        sett = cp_bn.get_all_variables()
        print(sett)

        while True:
            for variable, _ in cp_bn.get_all_edges():
                if variable in sett:
                    sett.remove(variable)
                    print(sett)

            for var in qe:
                if var in sett:
                    sett.remove(var)
            
            for item in sett:
                cp_bn.del_var(item)
            
            break

        # delete the edge outgoing from evidence variable e
        for variable in e:
            for child in cp_bn.get_children(variable):
                cp_bn.del_edge((variable, child))
        
        cp_bn.draw_structure()
        
    
q = ['bowel-problem', 'dog-out', 'light-on']
e = ['family-out']

dogproblem = BayesNet()
dogproblem.load_from_bifxml('testing/dog_problem.BIFXML')
reasoner = BNReasoner(dogproblem)
reasoner.prune(q, e)

# if variable == edge[0]
#                 if variable not in qe:
#                     for edge in bn.get_all_edges():
#                         print(edge)
#                         if variable != edge[0]:
#                             try:
#                                 bn.del_var(variable)
#                             except:
#                                 pass