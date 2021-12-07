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

        cp_bn.draw_structure()
        
        # delete the leaf nodes not in qe until there are no more lead nodes to delete
        while True:
            sett = cp_bn.get_all_variables() # a 'sett' to act as place to stores variables to be deleted
    
            # here I get rid of the variables that are not leaf nodes from sett
            for variable, _ in cp_bn.get_all_edges():
                if variable in sett:
                    sett.remove(variable)

            # here I remove any leaf nodes that are part of qe
            for var in qe:
                if var in sett:
                    sett.remove(var)

            if len(sett) == 0:
                break
            
            # here I delete what ever is left in sett from the BN
            for item in sett:
                cp_bn.del_var(item)
                print(cp_bn.get_all_variables())
           

        # delete the edge outgoing from evidence variable e
        for variable in e:
            for child in cp_bn.get_children(variable):
                cp_bn.del_edge((variable, child))
        
        cp_bn.draw_structure()
        
    
q = ['bowel-problem', 'dog-out']
e = ['family-out']

dogproblem = BayesNet()
dogproblem.load_from_bifxml('testing/dog_problem.BIFXML')
reasoner = BNReasoner(dogproblem)
reasoner.prune(q, e)
