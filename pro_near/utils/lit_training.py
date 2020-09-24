import copy
import torch
import torch.nn as nn
import dsl


from utils.data import pad_minibatch, unpad_minibatch, flatten_tensor, flatten_batch
from utils.logging import log_and_print


import os
import pytorch_lightning as pl

class LitModel(pl.LightningModule):

    #from dsl/neural_functions/feedfwd module
    def __init__(self, input_size, output_size, num_units):
        super(FeedForwardModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = num_units
        self.first_layer = nn.Linear(self.input_size, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, current_input):
        assert isinstance(current_input, torch.Tensor)
        current_input = current_input.to(device)
        current = F.relu(self.first_layer(current_input))
        current = self.out_layer(current)
        return current

    #from training.py
    
    def configure_optimizers(self, program, optimizer, lr):
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
                    queue.append(functionclass) #todo so for an incomplete program, are they already defined as neural functions?
        curr_optim = optimizer(all_params, lr)
        return curr_optim

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result