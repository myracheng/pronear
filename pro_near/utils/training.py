import copy
import torch
import torch.nn as nn
import dsl


from utils.data import pad_minibatch, unpad_minibatch, flatten_tensor
from utils.logging import log_and_print,print_program
from pprint import pprint
from cpu_unpickle import traverse, CPU_Unpickler
# from
import os
import pickle

os.environ['PYTHONHASHSEED']='0'
torch.manual_seed(0)
# np.random.seed(0)


def init_optimizer(program, optimizer, lr):
    queue = [program]
    all_params = []
    while len(queue) != 0:
        current_function = queue.pop()
        if issubclass(type(current_function), dsl.HeuristicNeuralFunction):
            current_function.init_model()
            all_params.append({'params' : current_function.model.parameters(),'lr' : lr})
        elif current_function.has_params:
            current_function.init_params()
            all_params.append({'params': list(current_function.parameters.values()), 'lr': lr})
        else:
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
    curr_optim = optimizer(all_params, lr)
    return curr_optim

def process_batch(program, batch, output_type, output_size, device='cpu'):
    # print(device)
    batch_input = [torch.tensor(traj) for traj in batch]
    batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
    batch_padded = batch_padded.to(device)
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

def change_key(d, l, full_tree_ind, new_value):
    for key,val in d.items(): 

        l.append(val)
        if len(l) == full_tree_ind+1:
            d[key] = new_value
            return
        try:
            if val.submodules is not None:
                change_key(val.submodules,l, full_tree_ind,new_value) 
        except AttributeError:
            continue

def execute_and_train_with_full(base_program_name, hole_node_ind, program, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, print_every=60):
    #load program
    # pprint(type(hole_node))
    # level_to_replace = hole_node[1]
    if device == 'cpu':
        base_program = CPU_Unpickler(open("%s.p" % base_program_name, "rb")).load()
    else:
        base_program = pickle.load(open("%s.p" % base_program_name, "rb"))

    curr_level = 0
    l = []
    traverse(base_program.submodules,l)
    # pprint(l)
    curr_program = base_program.submodules
    # print(program)
    # print(program.submodules)
    # pprint
    
    change_key(base_program.submodules, [], hole_node_ind, program.submodules["program"]) #should we just replace with program?
    log_and_print(print_program(base_program))
    # pickle.dump(base_program, open("neursym.p", "wb"))

    return execute_and_train(base_program, program, validset, trainset, train_config, output_type, output_size, neural, device)

def execute_and_train(base_program, program, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, print_every=60):
    # print('enter training initial')
    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']

    num_epochs = neural_epochs if neural else symbolic_epochs

    # initialize optimizer
    curr_optim = init_optimizer(program, optimizer, lr)

    # prepare validation set
    validation_input, validation_output = map(list, zip(*validset))
    validation_true_vals = torch.flatten(torch.stack(validation_output)).float().to(device)	
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}
    original_output_type = base_program.program.output_type
    original_output_size = base_program.program.output_size

    losses = []
    training_f1 = []

    for epoch in range(1, num_epochs+1):
        temp_l = 0
        # temp_f = 0
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.flatten(torch.stack(batch_output)).float().to(device)
            predicted_vals = process_batch(base_program, batch_input, original_output_type, original_output_size, device) #fix lol
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()
            # print(predicted_vals.shape, true_vals.shape)
            loss = lossfxn(predicted_vals, true_vals)
            training_metric, _ = evalfxn(predicted_vals, true_vals, num_labels=num_labels)
            # print('tutu metric')
            temp_l += float(loss.data)
            # temp_f += training_metric
            # print(training_metric)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()

            # if batchidx % print_every == 0 or batchidx == 0:
            #     log_and_print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

        # check score on validation set
        losses.append(temp_l/len(trainset))
        # training_f1.append(temp_f/len(trainset))
        with torch.no_grad():
            predicted_vals = process_batch(base_program, validation_input, original_output_type, original_output_size, device)
            metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)

        if use_valid_score:
            if metric < best_metric:
                best_program = copy.deepcopy(program)
                best_metric = metric
                best_additional_params = additional_params
        else:
            best_program = copy.deepcopy(program)
            best_metric = metric
            best_additional_params = additional_params

    # select model with best validation score
    program = copy.deepcopy(best_program)
    log_and_print("Training loss is: {:.4f}".format(loss))
    log_and_print("Validation score is: {:.4f}".format(best_metric))
    log_and_print("Average f1-score is: {:.4f}".format(1 - best_metric))
    log_and_print("Hamming accuracy is: {:.4f}".format(best_additional_params['hamming_accuracy']))
    
    return best_metric, losses, training_f1

def execute_and_train_og(base_program_name, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, print_every=60):

    if device == 'cpu':
        program = CPU_Unpickler(open("%s.p" % base_program_name, "rb")).load()
    else:
        program = pickle.load(open("%s.p" % base_program_name, "rb"))

    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']

    num_epochs = neural_epochs if neural else symbolic_epochs

    # initialize optimizer
    curr_optim = init_optimizer(program, optimizer, lr)

    # prepare validation set
    validation_input, validation_output = map(list, zip(*validset))
    validation_true_vals = torch.flatten(torch.stack(validation_output)).float().to(device)	
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    for epoch in range(1, num_epochs+1):
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.flatten(torch.stack(batch_output)).float().to(device)
            predicted_vals = process_batch(program, batch_input, output_type, output_size, device)
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()
            # print(predicted_vals.shape, true_vals.shape)
            loss = lossfxn(predicted_vals, true_vals)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()

            # if batchidx % print_every == 0 or batchidx == 0:
            #     log_and_print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

        # check score on validation set
        with torch.no_grad():
            predicted_vals = process_batch(program, validation_input, output_type, output_size, device)
            metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)

        if use_valid_score:
            if metric < best_metric:
                best_program = copy.deepcopy(program)
                best_metric = metric
                best_additional_params = additional_params
        else:
            best_program = copy.deepcopy(program)
            best_metric = metric
            best_additional_params = additional_params

    # select model with best validation score
    program = copy.deepcopy(best_program)
    log_and_print("Validation score is: {:.4f}".format(best_metric))
    log_and_print("Average f1-score is: {:.4f}".format(1 - best_metric))
    log_and_print("Hamming accuracy is: {:.4f}".format(best_additional_params['hamming_accuracy']))
    
    return best_metric, program

#mcheng substitute nonterminals with NNs