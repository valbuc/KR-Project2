import random

class Node:
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.cpt = []

def generate_DAG(fname, num_levels=3, node_per_level=3):

    nodes = []

    # Create Nodes
    for level in range(num_levels):
        current_level_nodes = []
        for node in range(node_per_level):
            current_level_nodes.append(Node('l' + str(level) + 'n' + str(node)))
        nodes.append(current_level_nodes)

    # Create Connections
    for l, current_level_nodes in enumerate(nodes):
        if l != len(nodes)-1:  #exclude last level
            for n, node in enumerate(current_level_nodes):
                next_level_nodes = nodes[l+1]
                num_children = random.randint(0, len(next_level_nodes)//2) #num children between 1 and num nodes in next level
                children = random.sample(range(len(next_level_nodes)), num_children)
                for child in children:
                    node.children.append(child)
                    next_level_nodes[child].parents.append(n)

    # Create CPTs
    for l, current_level_nodes in enumerate(nodes):
        for n, node in enumerate(current_level_nodes):
            num_parents = len(node.parents)
            for row in range(2 ** num_parents):
                p = round(random.uniform(0, 1), 2)
                node.cpt.append([p, round(1-p, 2)])

    # save to bifxml file
    with open(fname, 'w') as outfile:
        # header
        outfile.writelines('<?xml version="1.0" encoding="US-ASCII"?>\n<BIF VERSION="0.3">\n<NETWORK>\n<NAME>Random-BN</NAME>\n\n<!-- Variables -->\n')

        #nodes
        for l, current_level_nodes in enumerate(nodes):
            for n, node in enumerate(current_level_nodes):
                outfile.writelines('<VARIABLE TYPE="nature">\n')
                outfile.writelines('\t<NAME>' + str(node.name) + '</NAME>\n\t<PROPERTY> </PROPERTY>\n\t<OUTCOME>True</OUTCOME>\n\t<OUTCOME>False</OUTCOME>\n')
                outfile.writelines('</VARIABLE>\n\n')

        # CPTs
        outfile.writelines('\n<!-- Probability distributions -->\n')
        for l, current_level_nodes in enumerate(nodes):
            for n, node in enumerate(current_level_nodes):
                outfile.writelines('<DEFINITION>\n\t<FOR>'+ str(node.name) +'</FOR>\n')
                for parent in node.parents:
                    outfile.writelines('\t<GIVEN>' + str(nodes[l-1][parent].name) + '</GIVEN>\n')
                outfile.writelines('\t<TABLE>\n')
                for line in node.cpt:
                    outfile.writelines('\t\t' + str(line[0]) + ' ' + str(line[1]) + '\n')
                outfile.writelines('\t</TABLE>\n</DEFINITION>\n')
        outfile.writelines('\n</NETWORK>\n</BIF>')



if __name__ == '__main__':
    #num nodes = num_levels * node_per_level
    generate_DAG('test1.BIFXML', num_levels=6, node_per_level=6)