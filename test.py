from BayesNet import BayesNet
from taylor import BNReasoner
import networkx as nx
import copy

"""
#TD using this file to test code from Task1 (problems running)
"""
#TD: If at some point you get a "omp Error #15," try running this code. It worked for me, but apparently it is not the best solution as it can cause problems, so use with "caution" I suppose 
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dogproblem = BayesNet()
dogproblem.load_from_bifxml('testing/dog_problem.BIFXML')

reasoner = BNReasoner(dogproblem)

reasoner.ordering_mindegree()