import argparse
import os
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


def parse_args():
    parser = argparse.ArgumentParser()
    # Args for experiment setup
    parser.add_argument('-t', '--trial', type=int, required=True,
                        help="trial ID")
    parser.add_argument('--exp_name', type=str, required=True,
                        help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # Args for data
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
    parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
                        help="input type of data")
    parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
                        help="output type of data")
    parser.add_argument('--input_size', type=int, required=True,
                        help="dimenion of features of each frame")
    parser.add_argument('--output_size', type=int, required=True,
                        help="dimension of output of each frame (usually equal to num_labels")
    parser.add_argument('--num_labels', type=int, required=True,
                        help="number of class labels")

    # Args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="min number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.01,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")

    # Args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=0.8,
                        help="split training set for validation." +
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50,
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=6,
                        help="training epochs for symbolic programs")
    parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits"],
                        help="loss function for training")
    parser.add_argument('--class_weights', type=str, required=False, default=None,
                        help="weights for each class in the loss function, comma separated floats")
    parser.add_argument('--num_iter', type=int, required=False, default=10,
                        help="number of iterations for propel")
    parser.add_argument('--num_f_epochs', type=int, required=False, default=100,
                        help="length of training for the neural model")
    # Args for algorithms
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=["mc-sampling", "enumeration",
                            "genetic", "astar-near", "iddfs-near", "rnn"],
                        help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR" +
                        " (>1.0 prunes aggressively in this case)")
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")

    # parser.add_argument('--from_saved', type=bool, required=False, default=False,
    #                     help="load model from saved")
    
    parser.add_argument('--exp_id', type=int, required=False, default=None, help="experiment id")

    # parser.add_argument()
    return parser.parse_args()


class Propel():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        


        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        # load input data
        self.train_data = np.load(self.train_data)
        self.test_data = np.load(self.test_data)
        # print(train_data.shape)
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

        if self.exp_id is not None:
            # self.program_path #todo
            self.timestamp = self.exp_id
            full_exp_name = "{}_{}_{:03d}_{}".format(
                self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run
            self.save_path = os.path.join(self.save_dir, full_exp_name)
            program_iters =[f.split('_')[-1][:-2] for f in glob.glob(os.path.join(self.save_path,'*.p'))]
            program_iters.sort()
            self.curr_iter = int(program_iters[-1])
            # log_and_print(program_iters)
            # Load
            try:
                self.model = torch.load(os.path.join(self.save_path, "neural_model.pt"))
            except FileNotFoundError:
                log_and_print("no saved model found")
                self.model = self.init_neural_model(self.batched_trainset)

        # self.num_units = 240 #todo fix
        else:
            self.curr_iter = 0
            self.program_path = None 

            now = datetime.now()
            self.timestamp = str(datetime.timestamp(now)).split('.')[0]
            

            full_exp_name = "{}_{}_{:03d}_{}".format(
                self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run

            self.save_path = os.path.join(self.save_dir, full_exp_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            # load initial NN
            self.model = self.init_neural_model(self.batched_trainset)

    def run_propel(self):
        for i in range(self.curr_iter, self.num_iter):
            # log_and_print('Iteration %d' % i)
            self.run_near(self.model, i)
            self.update_f()
             # save model
            model_path = os.path.join(self.save_path, "neural_model.pt") #should we save every one?
            torch.save(self.model, model_path)

            self.evaluate()
        self.evaluate_composed()

    def evaluate_composed(self):
        #evaluate all of them together
        programs = []
        # for program in walkdir:
        # assert os.path.isfile(self.program_path)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            program = pickle.load(open(os.path.join(self.save_path, "program_0.p"), "rb"))
            ensemble_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            for i in range(1, self.num_iter):
                program = pickle.load(open(os.path.join(self.save_path, "program_%d.p" % i), "rb"))
                predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
                ensemble_vals += predicted_vals

            metric, additional_params = label_correctness(ensemble_vals/self.num_iter, true_vals, num_labels=self.num_labels)
        log_and_print("Ensemble F1 score achieved is {:.4f}".format(1 - metric))

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
            for batchidx in range(len(trainset)):	
                batch_input, batch_output = map(list, zip(*trainset[batchidx]))	
                start = time.time()
                true_vals = torch.flatten(torch.stack(batch_output)).float().to(self.device)	
                end = time.time()
                log_and_print('tutu2 %f' % (end - start))
                predicted_vals = self.process_batch(model_wrap, batch_input, self.output_type, self.output_size, self.device)	
                true_vals = true_vals.long()	
                loss = lossfxn(predicted_vals, true_vals)	
                optimizer.zero_grad()	
                loss.backward()
                optimizer.step() 	
            loss_values.append(loss.item())	
            if epoch % 50 == 0:	
                plt.plot(range(epoch),loss_values)	
                plt.savefig(os.path.join(self.save_path,'init_loss.png'))

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
            for batchidx in range(len(trainset)):	
                batch_input, batch_output = map(list, zip(*trainset[batchidx]))	
                true_vals = torch.flatten(torch.stack(batch_output)).float().to(self.device)	
                predicted_vals = self.process_batch(model_wrap, batch_input, self.output_type, self.output_size, self.device)	
                with torch.no_grad():	
                    program_vals = self.process_batch(program, batch_input, self.output_type, self.output_size, self.device)	
                # TODO a little hacky, but easiest solution for now	
                # if isinstance(lossfxn, nn.CrossEntropyLoss):	
                true_vals = true_vals.long()	
                #print(predicted_vals.shape, true_vals.shape)	
                loss = lossfxn((alpha * predicted_vals + (1 - alpha) * program_vals), true_vals)	
                # loss = lossfxn(predicted_vals, true_vals) + lossfxn(program_vals, true_vals) #second one is from program	
                optimizer.zero_grad()	
                loss.backward()	
                optimizer.step()  	
            loss_values.append(loss.item())	
            if epoch % 50 == 0:	
                plt.plot(range(epoch),loss_values)	
                plt.savefig(os.path.join(self.save_path,'loss_%s.png' % (self.program_path.split('/')[-1])))
if __name__ == '__main__':
    args = parse_args()
    propel_instance = Propel(**vars(args))
    propel_instance.run_propel()