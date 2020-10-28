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
import matplotlib.pyplot as plt	
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
from propel import parse_args

class Split_data():

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
        assert self.train_data.shape[-1] == self.test_data.shape[-1] == self.input_size
        if self.valid_data is not None and self.valid_labels is not None:
            self.valid_data = np.load(self.valid_data)
            self.valid_labels = np.load(self.valid_labels)
            assert valid_data.shape[-1] == self.input_size

        

        self.batched_trainset, self.validset, self.testset = prepare_datasets(self.train_data, self.valid_data, self.test_data, self.train_labels, self.valid_labels, 
        self.test_labels, normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)
        
        # assert os.path.isfile(self.program_path)
        # results/crim13_astar-near_001_1601734252/program_0.p
        if self.device == 'cpu':
            self.base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            self.base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        
        data = self.base_program.submodules
        l = []
        traverse(data,l)
        # print(l)
        self.hole_node = l[self.hole_node_ind]
        # random.choice(l)
        # pprint(self.hole_node[0].input_size)
        print('chosen hole')
        print(self.hole_node)
        
        #for near on subtree
        self.curr_iter = 0
        self.program_path = None 

        now = datetime.now()
        self.timestamp = str(datetime.timestamp(now)).split('.')[0]
        log_and_print(self.timestamp)

        full_exp_name = "{}_{}_{:03d}_{}".format(
            self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run

        self.save_path = os.path.join(self.save_dir, full_exp_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # load initial NN
        # self.model = self.init_neural_model(self.batched_trainset)
        # log_and_print('hi2')
        # self.evaluate()
        self.run_near()


    def evaluate(self):

        # assert os.path.isfile(self.program_path)
        program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()

        program_baby = CPU_Unpickler(open("results/crim13_astar-near_001_1603639887/program_0.p", "rb")).load()
        # program_baby = CPU_Unpickler(open("results/crim13_astar-near_001_1603682250/program_0.p", "rb")).load() #og lbabels
        

        # program_baby = CPU_Unpickler(open("results/crim13_astar-near_001_1603683071/program_0.p", "rb")).load() #F = 0.25
        data = program.submodules
        l = []
        traverse(data,l)
        # print(l)
        hole_node = l[self.hole_node_ind] #conditoin node
        # print(hole_node)
        change_key(program.submodules, hole_node[0], program_baby, hole_node[1]) 
        pickle.dump(program, open("ite_1603639887.p", "wb"))
        # program = pickle.load(open(self.program_path, "rb"))
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
        # log_and_print("Addition
        # al performance parameters: {}\n".format(additional_params))

    def run_near(self): 

        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(), #todo
            'evalfxn' : label_correctness,
            'num_labels' : self.num_labels
        }


        near_input_type = self.hole_node[0].input_type
        near_output_type = self.hole_node[0].output_type
        near_input_size = self.hole_node[0].input_size
        near_output_size = self.hole_node[0].output_size
        

        # Initialize program graph starting from trained NN
        program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS, near_input_type, near_output_type, near_input_size, near_output_size,
            self.max_num_units, self.min_num_units, self.max_num_children, self.max_depth, self.penalty, ite_beta=self.ite_beta)

        # Initialize algorithm
        algorithm = ASTAR_NEAR(frontier_capacity=self.frontier_capacity)
        best_programs = algorithm.run(self.base_program_name, self.hole_node,
            program_graph, self.batched_trainset, self.validset, train_config, self.device)

        if self.algorithm == "rnn":
            # special case for RNN baseline
            best_program = best_programs
        else:
            # Print all best programs found
            log_and_print("\n")
            log_and_print("BEST programs found:")
            for item in best_programs:
                print_program_dict(item)
            best_program = best_programs[-1]["program"]

        # Save best program
        num_iter = 0
        self.program_path = os.path.join(self.save_path, "program_%d.p" % num_iter)
        pickle.dump(best_program, open(self.program_path, "wb"))

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


    def init_neural_model(self, trainset):	
        #todo why is this so slow for crim 13	
        num_labels = self.num_labels	
        loss_values = []	
        # model	
        model_wrap = dsl.ListToListModule(	
            self.input_size, self.output_size, self.max_num_units)	
        lossfxn = nn.CrossEntropyLoss()	
        optimizer = optim.SGD(model_wrap.model.parameters(), lr=0.001, momentum=0.9)	
        num_epochs = self.num_f_epochs	
        for epoch in range(1, num_epochs+1):	
            # log_and_print(epoch)	
            batch_loss = 0
            for batchidx in range(len(trainset)):	
                batch_input, batch_output = map(list, zip(*trainset[batchidx]))	
                true_vals = torch.flatten(torch.stack(batch_output)).float().to(self.device)
                predicted_vals = self.process_batch(model_wrap, batch_input, self.output_type, self.output_size, self.device)	
                true_vals = true_vals.long()	
                loss = lossfxn(predicted_vals, true_vals)	
                optimizer.zero_grad()	
                loss.backward()
                optimizer.step() 	
                batch_loss += loss.item()
            loss_values.append(batch_loss / len(trainset))
            if epoch % 50 == 0:	
                plt.plot(range(epoch),loss_values)	
                plt.savefig(os.path.join(self.save_path,'init_loss.png'))
                plt.close()

        return model_wrap
    
if __name__ == '__main__':
    args = parse_args()
    propel_instance = Split_data(**vars(args))
    # propel_instance.run_near()
