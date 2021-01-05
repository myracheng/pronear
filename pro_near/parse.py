import numpy as np
import os
base_dir = 'results/mars_an_astar-near_1_1605457200'
file = open(os.path.join(base_dir, 'log.txt'), mode = 'r', encoding = 'utf-8-sig')
lines = file.readlines()
file.close()
my_dict = {}
my_list = []
children = []

lines = lines[12:]
# print(lines[0])
for i, line in enumerate(lines):
    if line.startswith('INFO:root:Training child program: Start') and 'AtomToAtom' not in line:

        score = float(lines[i+3].split(' ')[-1])
        line = line[40:].split('(')
        for k in line:
            if ')' not in k:
                if k in my_dict:
                    my_dict[k].append(score)
                else:
                    my_dict[k] = [score]
# print(my_dict)
# print(my_dict.keys())
new_dict = {}
for k in my_dict:
    new_dict[k] = str(np.mean(my_dict[k]))[:4]
print(new_dict)
# print(list(new_dict.values()))
# np.save(os.path.join(base_dir, 'weights.npy'),new_dict)