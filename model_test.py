import argparse
import time
import torch
import torch.nn.functional as F
import utils
import tabulate
import bisect
import os
import sys
from functools import partial
import collections
import models
from data import get_data
#from qtorch.optim import OptimLP
from optim import OptimLP
from torch.optim import SGD
from quant import *

model_name = sys.argv[1]
epochs = int(sys.argv[2])
k_val = int(sys.argv[3])
num_classes = int(sys.argv[4])
rounding = 'stochastic'
lr_init = 0.1
wd = 1e-4

print(f"TESTING LOG FOR MODEL {model_name}, K VALUE {k_val}\n\n")

weight = BlockMinifloat(exp=2, man=5, tile=-1, flush_to_zero=False, k_exp= k_val)
activate = BlockMinifloat(exp=2, man=5, tile=-1, flush_to_zero=False, k_exp= k_val)
error = BlockMinifloat(exp=4, man=3, tile=-1, flush_to_zero=False, k_exp= k_val)
accuracy = BlockMinifloat(exp=-1, man=-1, tile=-1, flush_to_zero=False, k_exp= k_val)
grad = BlockMinifloat(exp=-1, man=-1, tile=-1, flush_to_zero=False, k_exp= k_val)
momentum = BlockMinifloat(exp=-1, man=-1, tile=-1, flush_to_zero=False, k_exp= k_val)

weight_quantizer = quantizer(forward_number=weight, forward_rounding=rounding)
grad_quantizer   = quantizer(forward_number=grad, forward_rounding=rounding)
momentum_quantizer = quantizer(forward_number=momentum, forward_rounding=rounding)
acc_quantizer = quantizer(forward_number=accuracy, forward_rounding=rounding)
acc_err_quant = lambda : Quantizer(activate, error, rounding, rounding)

model_cfg = getattr(models, model_name)
model_cfg.kwargs.update({"quant":acc_err_quant})
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

#setting model weights equal to 1
for param in model.parameters():
    param.data = nn.parameter.Parameter(torch.ones_like(param))

criterion = F.cross_entropy
optimizer = SGD(model.parameters(), lr = lr_init, momentum=0.9, weight_decay = wd)

optimizer = OptimLP(optimizer, weight_quant=weight_quantizer, grad_quant=grad_quantizer, momentum_quant=momentum_quantizer, acc_quant=acc_quantizer)

for epoch in range(epochs):
    print(f"Training started, epoch: {epoch}\n")
    train_running_loss = 0.0

    data = torch.tensor([7.666, 6.98, 7.01, 0.00879, 0.0142, 0.0158])
    input = torch.reshape(data, (1,3,2))
    target = torch.ones(1)

    print("model weights: ", model.fc1.weight)

    optimizer.zero_grad()
    output = model(input)
    print("output: ", output)
    loss = criterion(output, target.long())
    print("loss: ", loss)
    loss.backward()
    print("optimiser step\n")
    optimizer.step()

    print("model weights after backward pass: ", model.fc1.weight)

    train_running_loss = loss.item()
    print(f"training loss: {train_running_loss}\n\n")

    