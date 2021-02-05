"""
PrGr.py:
"""
import networkx as nx
import numpy as np
import torch.utils.data as data
import os, sys
import argparse
import pickle

import datasets.utils

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from graph_reader import divide_datasets


class PrGr(data.Dataset):
    
    def __init__(self, root_path, ids, classes):
        
        self.root = root_path
        self.classes = classes
        self.ids = ids
        
    def __getitem__(self, index):
        #TODO: Manually have to check the convert_node_labels_to_integers function
        # print(self.ids[index])
        with open(os.path.join(self.root, self.ids[index]), 'rb') as t:                   
            G = pickle.load(t)
        g = nx.relabel.convert_node_labels_to_integers(G)
        # g = G
        target = self.classes[index]
        # print(target)

        h = self.vertex_transform(g)

        # e = 
        g, e = self.edge_transform(g)

        target = self.target_transform(target)
        # print(target)

        return (g, h, e), target
        
    def __len__(self):
        return len(self.ids)

    def vertex_transform(self, g):
        h = []
        fn_dict = {}
        count = 0
        for n, d in list(g.nodes(data=True)):
            h_t = []
            # print(d)
            fn_name = d['props'].name
            if fn_name in fn_dict:

                name_num = fn_dict[fn_name]
            else:
                fn_dict[fn_name] = count
                name_num = count
                count += 1
            h_t.append(name_num)
            # if d['props'].has_params:
            #     h_t.append(d['props'].parameters['weights'].data.numpy()) #tensor parameters
            # else:
            #     h_t.append('0')
            h.append(h_t)
        return h

    def edge_transform(self, g):
        e = {}
        for n1, n2, d in list(g.edges(data=True)):
            # print("hi")
            e_t = []
            e_t.append(1) #label for edge
            e[(n1, n2)] = e_t
        # print(e)
        return nx.to_numpy_matrix(g), e

    def target_transform(self, target):
        return [target]
    def set_target_transform(self, target_transform):
        self.target_transform = target_transform
    
if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='PrGr Object.')
    # Optional argument
    # 465684 is directed, 463726 isnt
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['../../pronear/pro_near/trees/results/465684'])
    args = parser.parse_args()
    root = args.root[0]
    
    label_file = 'labels.txt'
    list_file = 'graphs.txt'
    with open(os.path.join(root, label_file), 'r') as f:
        l = f.read()
        classes = [int(round(float(s))) for s in l.split()]#classes based on 0.5
        # print(set(classes))  
        unique, counts = np.unique(np.array(classes), return_counts=True)
        print(dict(zip(unique, counts)))
    with open(os.path.join(root, list_file), 'r') as f:

        files = [s + '.pkl' for s in f.read().splitlines()]
        
    train_ids, train_classes, valid_ids, valid_classes, test_ids, test_classes = divide_datasets(files, classes)

    data_train = PrGr(root, train_ids, train_classes)
    data_valid = PrGr(root, valid_ids, valid_classes)
    data_test = PrGr(root, test_ids, test_classes)
    
    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))
    
    print(data_train[1])
    print(data_valid[1])
    print(data_test[1])