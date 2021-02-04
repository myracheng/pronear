"""
Save programs as trees for use in GNNs
"""
from cpu_unpickle import CPU_Unpickler, traverse
import networkx as nx
import glob
import re
import os
import pickle
from datetime import datetime

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
        except AttrirebuteError:
            continue
            
root_dir = 'results/'
# root_dir = '../../../../../home/mccheng/near_programs/'
     
count = 0


now = datetime.now()
timestamp = str(datetime.timestamp(now)).split('.')[0][4:]
  

save_dir = os.path.join('trees',root_dir, timestamp)
os.makedirs(save_dir)


label_file = open("%s/labels.txt" % (save_dir),"w") 
for filename in glob.iglob(root_dir + '**/*.p', recursive=True):

    full_folder = os.path.dirname(filename)
    log_file = os.path.join(full_folder, 'log.txt')
    try:
        ### GET LABEL
        f = open(log_file, "r")
        lines = f.readlines()
        score = re.findall("is \d+\.\d+",lines[-1])[-1][3:]
        c = float(lines[-1][-7:-1]) #jank error checking for neurh.csv
        label_file.write("%s\n" % score)


        ### PROCESS PROGRAM
        folder = os.path.dirname(filename).split('/')[-1]
        # if count > 10: 
            # break
        count += 1
    
        base_program = CPU_Unpickler(open(filename, "rb")).load()
        data = base_program.submodules  
        G = nx.DiGraph()
        save_to_tree(data, G)
        
        with open('%s/%s.pkl'%(save_dir, folder), 'wb') as output:
            pickle.dump(G, output)
    except (FileNotFoundError, ValueError,IndexError) as e:
        #no log, incomplete log, etc
        print(e)

label_file.close()
        
print("Processed %d programs" % count)

### LOAD PROGRAMS
# a = 'trees/mars_an_astar-near_1_391458.pkl'                
# with open(a, 'rb') as t:                   
#   G = pickle.load(t) 


# for filename in glob.iglob(root_dir + '*/*.p', recursive=True):


### DEBUGGING 
# l = []
# traverse(data, l)
# print(l)

