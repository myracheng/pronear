"""
Save programs as trees for use in GNNs
"""
from cpu_unpickle import CPU_Unpickler, traverse
import networkx as nx
import glob
import os
import pickle

def save_to_tree(d, G):
    for key,val in d.items(): 
        G.add_node(val)
        try:
            if val.submodules is not None:
                kids = val.submodules.values()
                for k in kids:
                    G.add_node(k)
                    G.add_edge(val, k)
                save_to_tree(val.submodules,G) 
        except AttributeError:
            continue
            
root_dir = 'results/'
# root_dir = '../../../../../home/mccheng/near_programs/'
     
count = 0
for filename in glob.iglob(root_dir + '**/*.p', recursive=True):
    folder = os.path.dirname(filename).split('/')[-1]
    # print(folder)
    # if count >: 
        # break
    count += 1
   
    base_program = CPU_Unpickler(open(filename, "rb")).load()
    data = base_program.submodules  
    # l = []
    # traverse(data, l)
    # print(l)
    G = nx.DiGraph()
    save_to_tree(data, G)
    
    with open('trees/%s.pkl'%folder, 'wb') as output:
        pickle.dump(G, output)
        
print("Processed %d programs" % count)

# Load programs
# a = 'trees/mars_an_astar-near_1_391458.pkl'                
# with open(a, 'rb') as t:                   
#   G = pickle.load(t) 