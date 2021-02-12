#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodology defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]
"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import argparse
import os
import sys

# Our Modules
reader_folder = os.path.realpath(os.path.abspath('..'))
if reader_folder not in sys.path:
    sys.path.append(reader_folder)
import datasets
import numpy as np
from datasets import utils, PrGr
from models.MPNN_Duvenaud import MpnnDuvenaud
from LogMetric import AverageMeter, Logger
from graph_reader import read_2cols_set_files, create_numeric_classes,divide_datasets


torch.multiprocessing.set_sharing_strategy('file_system')


# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', default='PrGr', help='QM9')
# pronear/pro_near/trees/near_programs/474344
parser.add_argument('--datasetPath', default='../../pronear/pro_near/trees/near_programs/474344', help='dataset path')
# parser.add_argument('--datasetPath', default='../../pronear/pro_near/trees/results/463726', help='dataset path')
# parser.add_argument('--subSet', default='01_Keypoint', help='sub dataset')
parser.add_argument('--logPath', default='../log/f/duvenaud/', help='log path')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=13, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='Number of epochs to train (default: 360)')
parser.add_argument('--resume', default='test_dir',
                    help='directory of checkpoint')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')


def main():
    global args
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath

    print('Prepare files')
    
    label_file = 'labels.txt'
    list_file = 'graphs.txt'
    with open(os.path.join(root, label_file), 'r') as f:
        l = f.read()
        classes = [int(float(s) > 0.5) for s in l.split()]#classes based on 0.5
        # just makes them all 1
        # print(set(classes))  
        unique, counts = np.unique(np.array(classes), return_counts=True)
        print(dict(zip(unique, counts)))
    with open(os.path.join(root, list_file), 'r') as f:

        files = [s + '.pkl' for s in f.read().splitlines()]
        
    train_ids, train_classes, valid_ids, valid_classes, test_ids, test_classes = divide_datasets(files, classes)

    #shuffle here
    c = list(zip(train_ids, train_classes))

    random.shuffle(c)
    
    train_ids, train_classes = zip(*c)


    data_train = PrGr(root, train_ids, train_classes)
    print(data_train[0])
    print(len(data_train))
    data_valid = PrGr(root, valid_ids, valid_classes)
    data_test = PrGr(root, test_ids, test_classes)
    print(len(data_test))
    # Define model and optimizer
    print('Define model')
    # Select one graph
    g_tuple, l = data_train[6]
    g, h_t, e = g_tuple
    
    print('\tStatistics')
    stat_dict = datasets.utils.get_graph_stats(data_train, ['degrees'])

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=False, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                              num_workers=args.prefetch, pin_memory=True)
    criterion = nn.NLLLoss()
    evaluation = utils.accuracy
    print('\tCreate model')
    num_classes = 2
    print(stat_dict['degrees'])
    logger = Logger(args.logPath)

    model = torch.load('test.pth')
    print(model)
    return
    # print(model.r.learn_modules[0].fcs[3]) #penultimate layer
    model.eval()
    acc1 = validate_with_output(train_loader, model, criterion, evaluation, logger)
    print(acc1)
    print(train_classes)

    

def validate_with_output(val_loader, model, criterion, evaluation, logger=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if name in activation:
                activation[name] =  torch.cat([activation[name], output.detach()])
                # name.append(output.detah())
            else:
                # print(output.detach().shape)
                activation[name] = output.detach()
        return hook

    model.r.learn_modules[0].fcs[2].register_forward_hook(get_activation('fc2'))
    # x = torch.randn(1, 25)

    # switch to evaluate mode
    model.eval()

    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        target = torch.squeeze(target).type(torch.LongTensor)
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        test_loss = criterion(output, target)
        acc = Variable(evaluation(output.data, target.data, topk=(1,))[0])

        losses.update(test_loss.data, g.size(0))
        accuracies.update(acc.data, g.size(0))

    np.save('acs.npy', activation['fc2'].cpu().numpy())


    # return accuracies.avg


def validate(val_loader, model, criterion, evaluation, logger=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        target = torch.squeeze(target).type(torch.LongTensor)
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        test_loss = criterion(output, target)
        acc = Variable(evaluation(output.data, target.data, topk=(1,))[0])

        losses.update(test_loss.data, g.size(0))
        accuracies.update(acc.data, g.size(0))

    print(' * Average Accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(acc=accuracies, loss=losses))
          
    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_accuracy', accuracies.avg)

    return accuracies.avg

if __name__ == '__main__':
    main()
