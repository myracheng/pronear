"""
example command: 
python3 evaluate.py --train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy --test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy --valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy --base_program_name ite_1603639887 --baby_program_name 1603871744program_0
"""
# 1603905064
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
# import pytorch_lightning as pl
	
import time
from algorithms import ASTAR_NEAR, IDDFS_NEAR, MC_SAMPLING, ENUMERATION, GENETIC, RNN_BASELINE
# from dsl_current import DSL_DICT, CUSTOM_EDGE_COSTS
from dsl_crim13 import DSL_DICT, CUSTOM_EDGE_COSTS
# from eval import test_set_eval
from program_graph import ProgramGraph
from utils.data import *
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program_dict, log_and_print
import dsl
from utils.training import change_key
# from propel import parse_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True,
                        help="path to train data")
    parser.add_argument('--test_data', type=str, required=True,
                        help="path to test data")
    parser.add_argument('--valid_data', type=str, required=False, default=None,
                        help="path to val data. if this is not provided, we sample val from train.")
    parser.add_argument('--train_labels', type=str, required=True,
                        help="path to train labels")
    parser.add_argument('--test_labels', type=str, required=True,
                        help="path to test labels")
    parser.add_argument('--valid_labels', type=str, required=False, default=None,
                        help="path to val labels. if this is not provided, we sample val from train.")
    
    parser.add_argument('--train_valid_split', type=float, required=False, default=0.8,
                        help="split training set for validation." +
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50,
                        help="batch size for training set")
    parser.add_argument('--base_program_name', type=str, required=False, default="program_ite",
                        help="name of original program")
    parser.add_argument('--hole_node_ind', type=int, required=False, default="-1",
                        help="which node to replace")
    parser.add_argument('--baby_program_name', type=str, required=True, default="program_ite",
                        help="name of baby program")

    return parser.parse_args()

class Evaluate():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # load input data
        self.train_data = np.load(self.train_data)
        self.test_data = np.load(self.test_data)
        self.valid_data = None
        self.train_labels = np.load(self.train_labels)
        self.test_labels = np.load(self.test_labels)
        self.valid_labels = None
        if self.valid_data is not None and self.valid_labels is not None:
            self.valid_data = np.load(self.valid_data)
            self.valid_labels = np.load(self.valid_labels)

        

        self.batched_trainset, self.validset, self.testset = prepare_datasets(self.train_data, self.valid_data, self.test_data, self.train_labels, self.valid_labels, 
        self.test_labels, normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)
        
        if self.device == 'cpu':
            self.base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            self.base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        
        data = self.base_program.submodules
        l = []
        traverse(data,l)
        self.hole_node = l[self.hole_node_ind]
        
        #for near on subtree
        self.curr_iter = 0
        self.program_path = None 

        now = datetime.now()
        self.timestamp = str(datetime.timestamp(now)).split('.')[0]
        log_and_print(self.timestamp)

        self.evaluate()


    def evaluate(self):

        # assert os.path.isfile(self.program_path)
        base_program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()

        program_baby = CPU_Unpickler(open("%s.p" % self.baby_program_name, "rb")).load()
        data = base_program.submodules
        l = []
        traverse(data,l)
        # print(l)
        hole_node = l[self.hole_node_ind] #conditoin node
        # print(hole_node)
        change_key(base_program.submodules, hole_node[0], program_baby, hole_node[1]) 
        # pickle.dump(program, open("ite_1603639887.p", "wb"))
        base_output_type = base_program.program.output_type
        base_output_size = base_program.program.output_size
        # program = pickle.load(open(self.program_path, "rb"))
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(base_program, test_input, base_output_type, base_output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
    
    def process_batch(self, program, batch, output_type, output_size, device='cpu'):
        # batch_input = [torch.tensor(traj) for traj in batch]
        batch_input = torch.tensor(batch)
        batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
        batch_padded = batch_padded.to(device)
        # out_padded = program(batch_padded)
        out_padded = program.execute_on_batch(batch_padded, batch_lens)
        if output_type == "list":
            out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(program.output_type=='atom'))
        else:
            out_unpadded = out_padded
        if output_size == 1 or output_type == "list":
            return flatten_tensor(out_unpadded).squeeze()
        else:
            if isinstance(out_unpadded, list):
                out_unpadded = torch.cat(out_unpadded, dim=0).to(device)          
            return out_unpadded
    
if __name__ == '__main__':
    args = parse_args()
    propel_instance = Evaluate(**vars(args))
