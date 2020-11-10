"""
Sample command:
cd pronear/pro_near
python3.8 mars_search.py --algorithm astar-near --exp_name mars_an --trial 1 \
--train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz \
--valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz --test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz \
--train_labels "sniff" --input_type "list" --output_type "list" --input_size 316 --output_size 2 --num_labels 1 --lossfxn "crossentropy" \
--normalize --max_depth 3 --max_num_units 16 --min_num_units 6 --max_num_children 12 --learning_rate 0.001 --neural_epochs 8 --symbolic_epochs 15 \
 --base_program_name data/7keypoints/astar_1 --hole_node_ind 3 --penalty 0

1604788368pro_near/results/mars_an_astar-near_1_1604967353

pro_near/results/mars_an_astar-near_1_1604991739
"""
import argparse
import os
from cpu_unpickle import CPU_Unpickler, traverse
import pickle
import torch
import glob
import torch.nn as nn
from pprint import pprint
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from data import normalize_data, MyDataset
from datetime import datetime
	
import time
from algorithms import ASTAR_NEAR, IDDFS_NEAR, MC_SAMPLING, ENUMERATION, GENETIC, RNN_BASELINE
from dsl_mars import DSL_DICT, CUSTOM_EDGE_COSTS
from dsl.mars import MARS_INDICES
from program_graph import ProgramGraph
from utils.data import *
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program_dict, log_and_print,print_program
import dsl
from utils.training import change_key
from propel import parse_args

def preprocess(features, annotations, action, dct):
    new_features = []
    new_labels = []
    # Each feature array is (# of frames, # of features) dimensional
    for feat_arr, annot_arr in zip(features, annotations):
        length = len(feat_arr) - len(feat_arr) % 100
        # Instead of taking all the features, we only select the ones that are included in at least one feature subset
        feat_arr = np.take(feat_arr[:length,:], MARS_INDICES, axis = 1)
        feat_arr = feat_arr.reshape(-1, 100, len(MARS_INDICES))
        # if (np.isnan(np.sum(feat_arr))):
        #     print("F")
        feat_arr = np.where(np.logical_or(np.isnan(feat_arr), feat_arr == np.inf), 0, feat_arr).astype('float64')
        annot_arr = annot_arr[:length]

        label_arr = [1 if dct[x] == action else 0 for x in annot_arr]
        # print(label_arr)
        label_arr = np.asarray(label_arr).reshape(-1, 100).astype('float64')
        new_features.append(feat_arr)
        new_labels.append(label_arr)
    return np.concatenate(new_features, axis=0), np.concatenate(new_labels, axis=0)

def read_into_dict(fname):
    dct = {}
    with open(fname, 'r')  as file:
        for line in file:
            index = line.find(': ')
            key = line[:index].replace(' ', '_').strip()
            dct[key] = line[index+2:].strip()
    return dct
