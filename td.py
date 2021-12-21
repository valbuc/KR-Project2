from BayesNet import BayesNet
from BNReasoner import BNReasoner
import networkx as nx
import pandas as pd
import copy

# TD: If at some point you get a "omp Error #15," try running this code. It worked for me, but apparently it is not the best solution as it can cause problems, so use with "caution" I suppose 

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

covid = BayesNet()
covid.load_from_bifxml('testing/Use_Case.BIFXML')
reasoner = BNReasoner(covid)

##### What is the difference between the probability of someone going into ICU given that they have Covid-19, 
##### vs. the probability of someone going to the ICU given they took Ivermectin and they have Covid-19?

evidence = pd.Series({"Cold?": True})
query_vars = ["ICU?"]

marginals = reasoner.get_marginal_distribution('random', query_vars, evidence)
print(marginals)

evidence = pd.Series({"Cold?": True, "Take Ivermectin?": True})
query_vars = ["ICU?"]

marginalz = reasoner.get_marginal_distribution('random', query_vars, evidence)
print(marginalz)


### **MAP query**
##### What is the most likely instantiation of Covid-19 and ICU given that a person has cough, loss of smell, and a sore throat

map_vars = ['Covid-19?', 'ICU?']
e_vars = pd.Series({'Cough?': True, 'Loss of smell?': True, 'Sore throat?': True})

map_query = reasoner.MAP('random', map_vars=map_vars, e_vars=e_vars)

print(map_query)

### **MPE query**
##### What is the most likely instantiation of all variables given that a person tests positive on the PCR test and positive on the antigen test.

mpe_query = reasoner.MPE('mindegree', pd.Series({'Positive PCR test?': True, 'Positive antigen test?':True}))

print(mpe_query)