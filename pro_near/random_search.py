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
from dsl_crim13 import DSL_DICT, CUSTOM_EDGE_COSTS
from program_graph import ProgramGraph
from utils.data import *
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program_dict, log_and_print,print_program
import dsl
from utils.training import change_key
from propel import parse_args

class Split_data():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        print(self.__dict__.keys())
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
        print(l)
        self.hole_node = l[self.hole_node_ind]
        # random.choice(l)
        # pprint(self.hole_node[0].input_size)
        # print('chosen hole')
        print(self.hole_node)
        
        #for near on subtree
        self.curr_iter = 0
        self.program_path = None 

        now = datetime.now()
        self.timestamp = str(datetime.timestamp(now)).split('.')[0]
        log_and_print(self.timestamp)

        if self.exp_id is not None:
            self.trial = self.exp_id

        full_exp_name = "{}_{}_{}_{}".format(
            self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run

        self.save_path = os.path.join(self.save_dir, full_exp_name)
        if self.eval:
            self.evaluate()
        else:
            self.run_near()


    def evaluate(self):

        # assert os.path.isfile(self.program_path)
        program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()

        program_baby = CPU_Unpickler(open("results/crim13_astar-near_001_1603868377/program_0.p", "rb")).load()
        # program_baby = CPU_Unpickler(open("results/crim13_astar-near_001_1603682250/program_0.p", "rb")).load() #og lbabels
        

        # program_baby = CPU_Unpickler(open("results/crim13_astar-near_001_1601661498/program_0.p", "rb")).load() #F = 0.25
        data = program.submodules
        l = []
        traverse(data,l)
        # print(l)
        hole_node = l[self.hole_node_ind] #conditoin node
        print(hole_node)
        # l = []
        # traverse(program_baby.submodules,l)
        # print(l)
        change_key(program.submodules, hole_node[0], program_baby, hole_node[1]) 
        l = []
        traverse(program.submodules,l)
        print(l)
        # # pickle.dump(program, open("two_iter_other.p", "wb"))
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
        best_program_str = []
        if self.algorithm == "rnn":
            # special case for RNN baseline
            best_program = best_programs
        else:
            # Print all best programs found
            log_and_print("\n")
            log_and_print("BEST programs found:")
            for item in best_programs:
                program_struct = print_program(item["program"], ignore_constants=True)
                program_info = "struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
                    item["struct_cost"], item["score"], item["path_cost"], item["time"])
                best_program_str.append((program_struct, program_info))
                print_program_dict(item)
            best_program = best_programs[-1]["program"]

        # Save best programs
        f = open(os.path.join(self.save_path, "best_programs.txt"),"w")
        f.write( str(best_program_str) )
        f.close()

        # Save best program
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.program_path = os.path.join(self.save_path, "subprogram.p")
        pickle.dump(best_program, open(self.program_path, "wb"))

        self.full_path = os.path.join(self.save_path, "fullprogram.p")

        if self.device == 'cpu':
            base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        curr_level = 0
        l = []
        traverse(base_program.submodules,l)
        curr_program = base_program.submodules
        change_key(base_program.submodules, self.hole_node[0], best_program, self.hole_node[1])
        pickle.dump(base_program, open(self.full_path, "wb"))


        # Save parameters
        f = open(os.path.join(self.save_path, "parameters.txt"),"w")

        parameters = ['input_type', 'output_type', 'input_size', 'output_size', 'num_labels', 'neural_units', 'max_num_units', 
            'min_num_units', 'max_num_children', 'max_depth', 'penalty', 'ite_beta', 'train_valid_split', 'normalize', 'batch_size', 
            'learning_rate', 'neural_epochs', 'symbolic_epochs', 'lossfxn', 'class_weights', 'num_iter', 'num_f_epochs', 'algorithm', 
            'frontier_capacity', 'initial_depth', 'performance_multiplier', 'depth_bias', 'exponent_bias', 'num_mc_samples', 'max_num_programs', 
            'population_size', 'selection_size', 'num_gens', 'total_eval', 'mutation_prob', 'max_enum_depth', 'exp_id', 'base_program_name', 'hole_node_ind']
        for p in parameters:
            f.write( str(self.__dict__[p]) )
        f.close()

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
