"""
Sample command:

python3.8 mars_search.py --algorithm astar-near --exp_name mars_an --trial 1 \
--train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz \
--valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz --test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz \
--train_labels "sniff" --input_type "list" --output_type "list" --input_size 316 --output_size 2 --num_labels 1 --lossfxn "crossentropy" \
--normalize --max_depth 6 --max_num_units 16 --min_num_units 6 --max_num_children 6 --learning_rate 0.0005 --neural_epochs 8 --symbolic_epochs 15 \
--class_weights "1.0,1.0" --base_program_name data/7keypoints/astar_1
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
<<<<<<< HEAD
import random
=======
import randompython3.8 mars_search.py --algorithm astar-near --exp_name mars_an --trial 1 --train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz --valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz --test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz --train_labels "sniff" --input_type "list" --output_type "list" --input_size 316 --output_size 2 --num_labels 1 --lossfxn "crossentropy" --normalize --max_depth 6 --max_num_units 16 --min_num_units 6 --max_num_children 6 --learning_rate 0.0005 --neural_epochs 8 --symbolic_epochs 15 --class_weights "1.0,1.0" --base_program_name data/7keypoints/astar_1
>>>>>>> ab96c8f11099569f283b3e7e8f3565b3b6e3d06c
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
# 
class Subtree_search():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'


        #### start mars
        train_datasets = self.train_data.split(",")
        train_raw_features = []
        train_raw_annotations = []
        for fname in train_datasets:
            data = np.load(fname, allow_pickle=True)
            train_raw_features.extend(data["features"])
            train_raw_annotations.extend(data["annotations"])
        test_data = np.load(self.test_data, allow_pickle=True)

        test_raw_features = test_data["features"]
        test_raw_annotations = test_data["annotations"]
        valid_raw_features = None
        valid_raw_annotations = None
        valid_labels = None
        # Check the # of features of the first frame of the first video
        assert len(train_raw_features[0][0]) == len(test_raw_features[0][0]) == self.input_size

        if self.valid_data is not None:
            valid_data = np.load(self.valid_data, allow_pickle=True)
            valid_raw_features = valid_data["features"]
            valid_raw_annotations = valid_data["annotations"]
            assert len(valid_raw_features[0][0]) == self.input_size

        behave_dict = read_into_dict('../near_code_7keypoints/data/MARS_data/behavior_assignments_3class.txt')
        # Reshape the data to trajectories of length 100
        train_features, train_labels = preprocess(train_raw_features, train_raw_annotations, self.train_labels, behave_dict)
        test_features, test_labels = preprocess(test_raw_features, test_raw_annotations, self.train_labels, behave_dict)
        if valid_raw_features is not None and valid_raw_annotations is not None:
            valid_features, valid_labels = preprocess(valid_raw_features, valid_raw_annotations, self.train_labels, behave_dict)
        self.batched_trainset, self.validset, self.testset  = prepare_datasets(train_features, valid_features, test_features,
                                        train_labels, valid_labels, test_labels,
                                normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)

                            ##### END MARS
    
        if self.device == 'cpu':
            self.base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            self.base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        
        data = self.base_program.submodules
        l = []
        traverse(data,l)
        log_and_print(l)
        self.hole_node = l[self.hole_node_ind]

        # if self.hole_node_ind < 0:
            # self.hole_node_ind = len(l) + self.hole_node_ind
        #if negative, make it positive
        self.hole_node_ind %= len(l)
        
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
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            init_logging(self.save_path)
        
            self.run_near()


    
    def evaluate(self):
        program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        print(print_program(program, ignore_constants=True))
        l = []
        traverse(program.submodules,l)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))

    def run_near(self): 

        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1,0.9])), #todo
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
        best_programs = algorithm.run(self.timestamp, self.base_program_name, self.hole_node_ind,
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
        change_key(base_program.submodules, [], self.hole_node_ind, best_program)
        pickle.dump(base_program, open(self.full_path, "wb"))


        # Save parameters
        f = open(os.path.join(self.save_path, "parameters.txt"),"w")

        parameters = ['input_type', 'output_type', 'input_size', 'output_size', 'num_labels', 'neural_units', 'max_num_units', 
            'min_num_units', 'max_num_children', 'max_depth', 'penalty', 'ite_beta', 'train_valid_split', 'normalize', 'batch_size', 
            'learning_rate', 'neural_epochs', 'symbolic_epochs', 'lossfxn', 'class_weights', 'num_iter', 'num_f_epochs', 'algorithm', 
            'frontier_capacity', 'initial_depth', 'performance_multiplier', 'depth_bias', 'exponent_bias', 'num_mc_samples', 'max_num_programs', 
            'population_size', 'selection_size', 'num_gens', 'total_eval', 'mutation_prob', 'max_enum_depth', 'exp_id', 'base_program_name', 'hole_node_ind']
        for p in parameters:
            f.write( p + ': ' + str(self.__dict__[p]) + '\n' )
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


if __name__ == '__main__':
    args = parse_args()
    search_instance = Subtree_search(**vars(args))
from torch.utils.datapython3.8 mars_search.py --algorithm astar-near --exp_name mars_an --trial 1 --train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz --valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz --test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz --train_labels "sniff" --input_type "list" --output_type "list" --input_size 316 --output_size 2 --num_labels 1 --lossfxn "crossentropy" --normalize --max_depth 6 --max_num_units 16 --min_num_units 6 --max_num_children 6 --learning_rate 0.0005 --neural_epochs 8 --symbolic_epochs 15 --class_weights "1.0,1.0" --base_program_name data/7keypoints/astar_1
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
# 
class Subtree_search():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'


        #### start mars
        train_datasets = self.train_data.split(",")
        train_raw_features = []
        train_raw_annotations = []
        for fname in train_datasets:
            data = np.load(fname, allow_pickle=True)
            train_raw_features.extend(data["features"])
            train_raw_annotations.extend(data["annotations"])
        test_data = np.load(self.test_data, allow_pickle=True)

        test_raw_features = test_data["features"]
        test_raw_annotations = test_data["annotations"]
        valid_raw_features = None
        valid_raw_annotations = None
        valid_labels = None
        # Check the # of features of the first frame of the first video
        assert len(train_raw_features[0][0]) == len(test_raw_features[0][0]) == self.input_size

        if self.valid_data is not None:
            valid_data = np.load(self.valid_data, allow_pickle=True)
            valid_raw_features = valid_data["features"]
            valid_raw_annotations = valid_data["annotations"]
            assert len(valid_raw_features[0][0]) == self.input_size

        behave_dict = read_into_dict('../near_code_7keypoints/data/MARS_data/behavior_assignments_3class.txt')
        # Reshape the data to trajectories of length 100
        train_features, train_labels = preprocess(train_raw_features, train_raw_annotations, self.train_labels, behave_dict)
        test_features, test_labels = preprocess(test_raw_features, test_raw_annotations, self.train_labels, behave_dict)
        if valid_raw_features is not None and valid_raw_annotations is not None:
            valid_features, valid_labels = preprocess(valid_raw_features, valid_raw_annotations, self.train_labels, behave_dict)
        self.batched_trainset, self.validset, self.testset  = prepare_datasets(train_features, valid_features, test_features,
                                        train_labels, valid_labels, test_labels,
                                normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)

                            ##### END MARS
    
        if self.device == 'cpu':
            self.base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            self.base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        
        data = self.base_program.submodules
        l = []
        traverse(data,l)
        log_and_print(l)
        self.hole_node = l[self.hole_node_ind]

        # if self.hole_node_ind < 0:
            # self.hole_node_ind = len(l) + self.hole_node_ind
        #if negative, make it positive
        self.hole_node_ind %= len(l)
        
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
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            init_logging(self.save_path)
        
            self.run_near()


    
    def evaluate(self):
        program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        print(print_program(program, ignore_constants=True))
        l = []
        traverse(program.submodules,l)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))

    def run_near(self): 

        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1,0.9])), #todo
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
        best_programs = algorithm.run(self.timestamp, self.base_program_name, self.hole_node_ind,
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
        change_key(base_program.submodules, [], self.hole_node_ind, best_program)
        pickle.dump(base_program, open(self.full_path, "wb"))


        # Save parameters
        f = open(os.path.join(self.save_path, "parameters.txt"),"w")

        parameters = ['input_type', 'output_type', 'input_size', 'output_size', 'num_labels', 'neural_units', 'max_num_units', 
            'min_num_units', 'max_num_children', 'max_depth', 'penalty', 'ite_beta', 'train_valid_split', 'normalize', 'batch_size', 
            'learning_rate', 'neural_epochs', 'symbolic_epochs', 'lossfxn', 'class_weights', 'num_iter', 'num_f_epochs', 'algorithm', 
            'frontier_capacity', 'initial_depth', 'performance_multiplier', 'depth_bias', 'exponent_bias', 'num_mc_samples', 'max_num_programs', 
            'population_size', 'selection_size', 'num_gens', 'total_eval', 'mutation_prob', 'max_enum_depth', 'exp_id', 'base_program_name', 'hole_node_ind']
        for p in parameters:
            f.write( p + ': ' + str(self.__dict__[p]) + '\n' )
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


if __name__ == '__main__':
    args = parse_args()
    search_instance = Subtree_search(**vars(args)) import Dataset, DataLoader
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
# 
class Subtree_search():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'


        #### start mars
        train_datasets = self.train_data.split(",")
        train_raw_features = []
        train_raw_annotations = []
        for fname in train_datasets:
            data = np.load(fname, allow_pickle=True)
            train_raw_features.extend(data["features"])
            train_raw_annotations.extend(data["annotations"])
        test_data = np.load(self.test_data, allow_pickle=True)

        test_raw_features = test_data["features"]
        test_raw_annotations = test_data["annotations"]
        valid_raw_features = None
        valid_raw_annotations = None
        valid_labels = None
        # Check the # of features of the first frame of the first video
        assert len(train_raw_features[0][0]) == len(test_raw_features[0][0]) == self.input_size

        if self.valid_data is not None:
            valid_data = np.load(self.valid_data, allow_pickle=True)
            valid_raw_features = valid_data["features"]
            valid_raw_annotations = valid_data["annotations"]
            assert len(valid_raw_features[0][0]) == self.input_size

        behave_dict = read_into_dict('../near_code_7keypoints/data/MARS_data/behavior_assignments_3class.txt')
        # Reshape the data to trajectories of length 100
        train_features, train_labels = preprocess(train_raw_features, train_raw_annotations, self.train_labels, behave_dict)
        test_features, test_labels = preprocess(test_raw_features, test_raw_annotations, self.train_labels, behave_dict)
        if valid_raw_features is not None and valid_raw_annotations is not None:
            valid_features, valid_labels = preprocess(valid_raw_features, valid_raw_annotations, self.train_labels, behave_dict)
        self.batched_trainset, self.validset, self.testset  = prepare_datasets(train_features, valid_features, test_features,
                                        train_labels, valid_labels, test_labels,
                                normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)

                            ##### END MARS
    
        if self.device == 'cpu':
            self.base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            self.base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        
        data = self.base_program.submodules
        l = []
        traverse(data,l)
        log_and_print(l)
        self.hole_node = l[self.hole_node_ind]

        # if self.hole_node_ind < 0:
            # self.hole_node_ind = len(l) + self.hole_node_ind
        #if negative, make it positive
        self.hole_node_ind %= len(l)
        
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
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            init_logging(self.save_path)
        
            self.run_near()


    
    def evaluate(self):
        program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        print(print_program(program, ignore_constants=True))
        l = []
        traverse(program.submodules,l)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))

    def run_near(self): 

        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1,0.9])), #todo
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
        best_programs = algorithm.run(self.timestamp, self.base_program_name, self.hole_node_ind,
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
        change_key(base_program.submodules, [], self.hole_node_ind, best_program)
        pickle.dump(base_program, open(self.full_path, "wb"))


        # Save parameters
        f = open(os.path.join(self.save_path, "parameters.txt"),"w")

        parameters = ['input_type', 'output_type', 'input_size', 'output_size', 'num_labels', 'neural_units', 'max_num_units', 
            'min_num_units', 'max_num_children', 'max_depth', 'penalty', 'ite_beta', 'train_valid_split', 'normalize', 'batch_size', 
            'learning_rate', 'neural_epochs', 'symbolic_epochs', 'lossfxn', 'class_weights', 'num_iter', 'num_f_epochs', 'algorithm', 
            'frontier_capacity', 'initial_depth', 'performance_multiplier', 'depth_bias', 'exponent_bias', 'num_mc_samples', 'max_num_programs', 
            'population_size', 'selection_size', 'num_gens', 'total_eval', 'mutation_prob', 'max_enum_depth', 'exp_id', 'base_program_name', 'hole_node_ind']
        for p in parameters:
            f.write( p + ': ' + str(self.__dict__[p]) + '\n' )
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


if __name__ == '__main__':
    args = parse_args()
    search_instance = Subtree_search(**vars(args))