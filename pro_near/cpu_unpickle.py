import pickle
import torch
import io
import dsl
import random
from pprint import pprint
from dsl_crim13 import DSL_DICT, CUSTOM_EDGE_COSTS

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# ...
#contents = pickle.load(f) becomes...


c = CPU_Unpickler(open("program_ite.p", "rb")).load()


# print(out)
data = c.submodules

def traverse(d,l,level = 0): 
    # print(d)
    for key,val in d.items(): 

        l.append((val, level)) 
        try:
            if val.submodules is not None:
                traverse(val.submodules,l, level+1) 
        except AttributeError:
            continue

def get_options(dsl_dict, node):
    for key,val in dsl_dict.items(): 
        if type(node) in val:

            print(type(node))

out=[]
# for d in data:
    # print(d)
l=[]
traverse(data,l)
# print(l[0][0])
random_node = random.choice(l)

def change_key(d, required_key, new_value):
    for k, v in d.items():
        if isinstance(v, dict):
            change_key(v, required_key, new_value)
        if k == required_key:
            d[k] = new_value

# c.program
# get_options(DSL_DICT,l[0][0])
# out.append(l)

# unpacked = [list(unpack(item)) for item in data]

# pprint(l)
# c = CPU_Unpickler(open("program_list_to_atom.p", "rb")).load()
# pprint(vars(c.submodules['program'].submodules['subfunction'].submodules['evalfunction']))
# 

# pprint(vars(c))
# print(type(contents.program))
# pprint(c.program)
# pprint(type(c.submodules['program']))

# pprint(vars(c.submodules['program']))

#find it todo
# while (type(c.subprogram) != dsl.library_functions.SimpleITE)
# print(type(c.submodules['program'].submodules['mapfunction']))
# <class 'dsl.library_functions.SimpleITE'>
# ['mapfunction']))