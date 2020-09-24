import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data import normalize_data, MyDataset
from datetime import datetime
import pytorch_lightning as pl

from algorithms import ASTAR_NEAR, IDDFS_NEAR, MC_SAMPLING, ENUMERATION, GENETIC, RNN_BASELINE
from dsl_current import DSL_DICT, CUSTOM_EDGE_COSTS
# from dsl_crim13 import DSL_DICT, CUSTOM_EDGE_COSTS
# from eval import test_set_eval
from program_graph import ProgramGraph
from utils.data import prepare_datasets
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program_dict, log_and_print
from neural_agent import *
from lit import LitClassifier


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
                        help="max number of hidden units for neural programs")
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

    return parser.parse_args()


class Propel():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.num_units = 240 #todo fix
        self.program_path = None #most
        # data
        # batched_trainset, validset, testset = prepare_datasets(train_data, valid_data, test_data, train_labels, valid_labels, 
        # test_labels, normalize=args.normalize, train_valid_split=args.train_valid_split, batch_size=args.batch_size)

        self.train_dataset, self.valid_dataset, self.test_dataset = self.extract_data()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        # load initial NN
        self.model = self.init_neural_model(train_loader)


    def extract_data(self):
        # load input data
        # print(self.train_data)
        train_data = np.load(self.train_data)
        test_data = np.load(self.test_data)
        # print(train_data.shape)
        valid_data = None
        train_labels = np.load(self.train_labels)
        test_labels = np.load(self.test_labels)
        valid_labels = None
        assert train_data.shape[-1] == test_data.shape[-1] == self.input_size

        if self.valid_data is not None and self.valid_labels is not None:
            valid_data = np.load(self.valid_data)
            valid_labels = np.load(self.valid_labels)
            assert valid_data.shape[-1] == self.input_size

        if self.normalize:
            train_data, valid_data, test_data = normalize_data(
                train_data, valid_data, test_data)
        print(train_data.shape)
        train_dataset = MyDataset(train_data, train_labels)
        if self.valid_data is not None and self.valid_labels is not None:
            valid_dataset = MyDataset(valid_data, valid_labels)
        else: 
            valid_dataset = None

        test_dataset = MyDataset(test_data, test_labels)

        return train_dataset, valid_dataset, test_dataset

    def load_data(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, valid_loader, test_loader

    def run_propel(self):
        for i in range(num_iter):
            run_near(model)
            update_f()
            evaluate()
            

    def run_near(self, model): 

        now = datetime.now()
        timestamp = str(datetime.timestamp(now)).split('.')[0]

        full_exp_name = "{}_{}_{:03d}_{}".format(
            self.exp_name, self.algorithm, self.trial, timestamp) #unique timestamp for each near run

        save_path = os.path.join(self.save_dir, full_exp_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_config = {
            'lr' : args.learning_rate,
            'neural_epochs' : args.neural_epochs,
            'symbolic_epochs' : args.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : lossfxn,
            'evalfxn' : label_correctness,
            'num_labels' : args.num_labels
        }

        # Initialize program graph starting from trained NN
        program_graph = ProgramGraph(model, DSL_DICT, CUSTOM_EDGE_COSTS, self.input_type, self.output_type, self.input_size, self.output_size,
            self.max_num_units, self.min_num_units, self.max_num_children, self.max_depth, self.penalty, ite_beta=self.ite_beta)

        # Initialize algorithm
        algorithm = ASTAR_NEAR(frontier_capacity=self.frontier_capacity)
        best_programs = algorithm.run(
            program_graph, batched_trainset, validset, train_config, device)

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
        program_path = os.path.join(save_path, "program.p")
        pickle.dump(best_program, open(program_path, "wb"))

    def init_neural_model(self, train_loader):

        # model
        model = LitClassifier(
            self.input_size, self.output_size, self.num_units)

        # training
        trainer = pl.Trainer(gpus=0, max_epochs=2,
                             limit_train_batches=200)  # todo fix gpus

        trainer.fit(model, train_loader)

        return model

        # trainer.test(test_dataloaders=test_dataset)

    def update_f(train_loader, valid_loader, program_path):  # do data loader be iterable/reusable?
        # get loss from program...
        alpha = 0.5  # todo make this changeable

        # Load program
        assert os.path.isfile(program_path)
        program = pickle.load(open(program_path, "rb"))

        # todo so we can do it directly on data... use pl.Trainer?
        program_output = program_path(train_loader)
        nn_output = trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    args = parse_args()
    propel_instance = Propel(algorithm=args.algorithm, train_data=args.train_data, valid_data=args.valid_data, test_data=args.test_data, 
        train_labels=args.train_labels, valid_labels=args.valid_labels, test_labels=args.test_labels, 
        frontier_capacity=args.frontier_capacity, batch_size=args.batch_size, input_type=args.input_type, output_type=args.output_type, 
        input_size=args.input_size, output_size=args.output_size,
        max_num_units=args.max_num_units, min_num_units=args.min_num_units, max_num_children=args.max_num_children, 
        max_depth=args.max_depth, penalty=args.penalty, ite_beta=args.ite_beta, save_dir = args.save_dir,
        exp_name=args.exp_name, trial=args.trial,
        learning_rate= args.learning_rate,
        neural_epochs=args.neural_epochs,
        symbolic_epochs=args.symbolic_epochs,
        num_labels=args.num_labels,normalize=args.normalize)
    # return None
