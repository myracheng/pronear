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
import torch

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

    parser.add_argument('--neural_units', type=int, required=False, default=100,
                        help="max number of hidden units for neural programs")
    

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

    parser.add_argument('--exp_id', type=int, required=False, default=None, help="experiment id")

    # parser.add_argument()
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()


    if torch.cuda.is_available():
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    # load input data
    args.train_data = np.load(args.train_data)
    args.test_data = np.load(args.test_data)
    # print(train_data.shape)
    args.valid_data = None
    args.train_labels = np.load(args.train_labels)
    args.test_labels = np.load(args.test_labels)
    args.valid_labels = None
    assert args.train_data.shape[-1] == args.test_data.shape[-1] == args.input_size
    if args.valid_data is not None and args.valid_labels is not None:
        args.valid_data = np.load(args.valid_data)
        args.valid_labels = np.load(args.valid_labels)
        assert valid_data.shape[-1] == args.input_size


    args.batched_trainset, args.validset, args.testset = prepare_datasets(args.train_data, args.valid_data, args.test_data, args.train_labels, args.valid_labels, 
    args.test_labels, normalize=args.normalize, train_valid_split=args.train_valid_split, batch_size=args.batch_size)

    programs = []
    TEST_FOLDER_NAME = 'results/crim13_astar-near_001_1601506435/*'#TODO
            # for program in walkdir:
            # assert os.path.isfile(args.program_path)

    program_iters =[f for f in glob.glob(TEST_FOLDER_NAME)]
    # program_iters.sort()
    # last_program_iter = int(program_iters[-1])

    with torch.no_grad():
        test_input, test_output = map(list, zip(*args.testset))
        true_vals = torch.flatten(torch.stack(test_output)).float().to(args.device)	
        program = torch.load(open(program_iters[0], "rb"),map_location=torch.device('cpu'))
        
        ensemble_vals = args.process_batch(program, test_input, args.output_type, args.output_size, args.device)
        for f in glob.glob(TEST_FOLDER_NAME)[1:]:
            program = torch.load(open(f, "rb"),map_location=torch.device('cpu'))
            predicted_vals = args.process_batch(program, test_input, args.output_type, args.output_size, args.device)
            ensemble_vals += predicted_vals

        metric, additional_params = label_correctness(ensemble_vals/args.num_iter, true_vals, num_labels=args.num_labels)
    log_and_print("Ensemble F1 score achieved is {:.4f}".format(1 - metric))
