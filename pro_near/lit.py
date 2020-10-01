# from argparse import ArgumentParser

# import torch
# # import pytorch_lightning as pl
# import torch.nn as nn
# # from pytorch_lightning.metrics.functional import accuracy
# from torch.nn import functional as F



# class LitClassifier(pl.LightningModule):
#     def __init__(self, input_size, output_size, num_units, learning_rate=1e-3, batch_size=32, num_workers=4, program=None,**kwargs):
#         super().__init__()
#         self.save_hyperparameters()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = num_units
#         self.l1 = nn.Linear(self.input_size, self.hidden_size)
#         self.l2 = nn.Linear(self.hidden_size, self.output_size)
#         self.program = program #model from the program

#     def forward(self, x):
#         # current_input = current_input.to(device)
#         current = F.relu(self.l1(x))
#         current = self.l2(current)
#         return current

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         if self.program is not None:
#             y_program = self.program(x)
#             loss = F.cross_entropy(y_hat, y) + F.cross_entropy(y_hat, y_program)
#         else:
#             lossfxn = nn.CrossEntropyLoss()

#             loss = lossfxn(y_hat, y)
#             # loss = F.cross_entropy(y_hat, y)
#         result = pl.TrainResult(minimize=loss)
#         result.log('train_loss', loss)
#         return result

#     # def validation_step(self, batch, batch_idx):
#     #     x, y = batch
#     #     y_hat = self(x)
#     #     loss = F.cross_entropy(y_hat, y)
#     #     result = pl.EvalResult(checkpoint_on=loss)
#     #     result.log('val_loss', loss)
#     #     result.log('val_acc', accuracy(y_hat, y))
#     #     return result

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = F.cross_entropy(y_hat, y)
#         result = pl.EvalResult(checkpoint_on=loss)
#         result.log('test_loss', loss)
#         result.log('test_acc', accuracy(y_hat, y))
#         return result

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
