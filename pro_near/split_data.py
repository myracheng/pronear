#load in program
#find every if/then/else
#load on training data: split the training data based on which one it goes to
# do NEAR on the part inside the if/then/else based on that subset of training data


# Evaluating program Start(Map(SimpleITE(AngleSelect(), DistanceSelect(), PositionSelect()))) on TEST SET
# F1 score achieved is 0.2298
# Additional performance parameters: {'hamming_accuracy': 0.8602946156451067, 'all_f1s': array([0.92317977, 0.229843  ])}

import argparse
import os
from cpu_unpickle import CPU_Unpickler
import pickle
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        self.base_program = CPU_Unpickler(open("program_ite.p", "rb")).load()

        # pickle.load(open("program_ite.p", "rb"))

        
        # map_output = self.submodules["mapfunction"].execute_on_batch(map_input)

        # open("program_ite.p", "rb")).load()
        self.ite = self.base_program.submodules['program'].submodules['mapfunction'].submodules['evalfunction'] #if condition
        trainset = self.batched_trainset
        print(len(trainset))
        for batchidx in range(len(trainset)):	
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))	
            # true_vals = torch.flatten(torch.stack(batch_output)).float().to(self.device)
            # predicted_vals = self.process_batch(model_wrap, batch_input, self.output_type, self.output_size, self.device)	
            # def process_batch(self, program, batch, output_type, output_size, device='cpu'):
            batch_input = torch.tensor(batch_input)
            batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
            batch_padded = batch_padded.to(self.device)
            # print(batch_padded.size())

            batch_size, seq_len, feature_dim = batch_padded.size()
            map_input = batch_padded.view(-1, feature_dim) #x, 19
            print(map_input.shape)

            out = torch.sigmoid(self.ite.execute_on_batch(map_input, batch_lens))
            out2 = out.view(batch_size, seq_len, -1) #50, 100, 1?
            a = out[out > 0.5]
            b = out[out <= 0.5]
            # print(out2.size())
            # print(b.size())
            # out_unpadded = out_padded
        # print(type(out_padded))

        # self.bsmooth = nn.Sigmoid()
        # gate = self.bsmooth(predicted_eval)
        # self.beta = beta
            # return flatten_tensor(out_unpadded).squeeze()
        # conditional mask on tensor
        # then remove all the 0 values?


        #find if then else
        #pass data through

        # if self.exp_id is not None:
        #     # self.program_path #todo
        #     self.timestamp = self.exp_id
        #     full_exp_name = "{}_{}_{:03d}_{}".format(
        #         self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run
        #     self.save_path = os.path.join(self.save_dir, full_exp_name)
        #     program_iters =[f.split('_')[-1][:-2] for f in glob.glob(os.path.join(self.save_path,'*.p'))]
        #     program_iters.sort()
        #     self.curr_iter = int(program_iters[-1])
        #     # log_and_print(program_iters)
        #     # Load
        #     try:
        #         self.model = torch.load(os.path.join(self.save_path, "neural_model.pt"))
        #         log_and_print(type(self.model))
        #         log_and_print("the saved model found")
        #     except FileNotFoundError:
        #         log_and_print("no saved model found")
        #         self.model = self.init_neural_model(self.batched_trainset)

        # else:
            # self.curr_iter = 0
            # self.program_path = None 

            # now = datetime.now()
            # self.timestamp = str(datetime.timestamp(now)).split('.')[0]
            # log_and_print(self.timestamp)

            # full_exp_name = "{}_{}_{:03d}_{}".format(
            #     self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run

            # self.save_path = os.path.join(self.save_dir, full_exp_name)
            # if not os.path.exists(self.save_path):
            #     os.makedirs(self.save_path)
            # # load initial NN
            # self.model = self.init_neural_model(self.batched_trainset)

    def run_propel(self):
        model_path = os.path.join(self.save_path, "neural_model.pt") #should we save every one?
        for i in range(self.curr_iter, self.num_iter):
            torch.save(self.model, model_path)
            self.run_near(self.model, i)
            self.update_f()
            self.evaluate()
            self.evaluate_composed()

    def evaluate(self):

        assert os.path.isfile(self.program_path)
        program = pickle.load(open(self.program_path, "rb"))
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
        # log_and_print("Addition
        # al performance parameters: {}\n".format(additional_params))

    def run_near(self, model, num_iter): 

        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(), #todo
            'evalfxn' : label_correctness,
            'num_labels' : self.num_labels
        }

        # Initialize program graph starting from trained NN
        program_graph = ProgramGraph(model, DSL_DICT, CUSTOM_EDGE_COSTS, self.input_type, self.output_type, self.input_size, self.output_size,
            self.max_num_units, self.min_num_units, self.max_num_children, self.max_depth, self.penalty, ite_beta=self.ite_beta)

        # Initialize algorithm
        algorithm = ASTAR_NEAR(frontier_capacity=self.frontier_capacity)
        best_programs = algorithm.run(
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
    
    def update_f(self): 	
        # get loss from program...	
        alpha = 0.5  # todo make this changeable weight	
        trainset = self.batched_trainset	
        # Load program	
        assert os.path.isfile(self.program_path)	
        program = pickle.load(open(self.program_path, "rb"))	
        	
        	
        # model_wrap = dsl.ListToListModule(	
        #     self.input_size, self.output_size, self.max_num_units)	
        model_wrap = self.model	
        lossfxn = nn.CrossEntropyLoss()	
        loss_values = []	
        optimizer = optim.SGD(model_wrap.model.parameters(), lr=0.001, momentum=0.9)	
        num_epochs = self.num_f_epochs #todo fix	
        for epoch in range(1, num_epochs+1):	
            batch_loss = 0
            for batchidx in range(len(trainset)):	
                batch_input, batch_output = map(list, zip(*trainset[batchidx]))	
                true_vals = torch.flatten(torch.stack(batch_output)).float().to(self.device)	
                predicted_vals = self.process_batch(model_wrap, batch_input, self.output_type, self.output_size, self.device)	
                with torch.no_grad():	
                    program_vals = self.process_batch(program, batch_input, self.output_type, self.output_size, self.device)	
                	
                true_vals = true_vals.long()	
                loss = lossfxn((alpha * predicted_vals + (1 - alpha) * program_vals), true_vals)	
                optimizer.zero_grad()	
                loss.backward()	
                optimizer.step()  	
                batch_loss += loss.item()
            loss_values.append(batch_loss / len(trainset))	
            if epoch % 50 == 0:	
                plt.plot(range(epoch),loss_values)	
                plt.savefig(os.path.join(self.save_path,'loss_%s.png' % (self.program_path.split('/')[-1])))
                plt.close()
if __name__ == '__main__':
    args = parse_args()
    propel_instance = Split_data(**vars(args))
    # propel_instance.run_propel()
