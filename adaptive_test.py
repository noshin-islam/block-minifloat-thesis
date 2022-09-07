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
# from swa_utils import AveragedModel, update_bn, SWALR
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
import pdb


model_name = sys.argv[1]
epochs = int(sys.argv[2])
# k_val = int(sys.argv[3])

num_classes = int(sys.argv[3])
rounding = 'stochastic'
lr_init = 0.1
wd = 1e-4

adaptive = False
k_val = 1
checkpoint_num = 5
PATH = 'testlogs/checkpoint.pth'

seed = 3
torch.manual_seed(seed)

##SWA
swa_flag = True
swa_lr = 0.05
swa_start_epoch = 5
swa_c_epochs=1

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

## setting up SWA
if swa_flag == True:
    print('SWA training')
    swa_model = None
else:
    print('SGD training')

#setting model weights equal to 1
for param in model.parameters():
    param.data = nn.parameter.Parameter(torch.ones_like(param))

criterion = F.cross_entropy
optimizer = SGD(model.parameters(), lr = lr_init, momentum=0.9, weight_decay = wd)

optimizer = OptimLP(optimizer, weight_quant=weight_quantizer, grad_quant=grad_quantizer, momentum_quant=momentum_quantizer, acc_quant=acc_quantizer)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
swa_scheduler = SWALR(optimizer, swa_lr= swa_lr)

for epoch in range(epochs):
    print(f"Training started, epoch: {epoch+1}\n")
    train_running_loss = 0.0
    data = torch.tensor([7.666, 6.98, 7.01, 0.00879, 0.0142, 0.0158])
    input = torch.reshape(data, (1,3,2))
    target = torch.ones(1)
 
    print("model weights before optimiser step: ", model.fc1.weight)

    if adaptive==True:
        if ((epoch+1) != 1 and weight.k_exp != 1 and (epoch+1) % checkpoint_num == 0):
            print("K value switch")
            # weight activate error accuracy grad momentum
            print(f"Old k = {weight.k_exp}")

            weight.change_k(weight.k_exp-1)
            activate.change_k(activate.k_exp-1)
            error.change_k(error.k_exp-1)
            accuracy.change_k(accuracy.k_exp-1)
            grad.change_k(grad.k_exp-1)
            momentum.change_k(momentum.k_exp-1)

            print(f"New k = {weight.k_exp}")

            weight_quantizer = quantizer(forward_number=weight, forward_rounding=rounding)
            grad_quantizer   = quantizer(forward_number=grad, forward_rounding=rounding)
            momentum_quantizer = quantizer(forward_number=momentum, forward_rounding=rounding)
            acc_quantizer = quantizer(forward_number=accuracy, forward_rounding=rounding)
            acc_err_quant = lambda : Quantizer(activate, error, rounding, rounding)

            model_cfg = getattr(models, model_name)
            model_cfg.kwargs.update({"quant":acc_err_quant})
            model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            print("model weights before loading checkpoint: ", model.fc1.weight)
            ## loading back the model state dictionary into the newly defined model
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])

            print("model weights after loading checkpoint: ", model.fc1.weight)


            criterion = F.cross_entropy
            optimizer = SGD(model.parameters(), lr = lr_init, momentum=0.9, weight_decay = wd)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            optimizer = OptimLP(optimizer, weight_quant=weight_quantizer, grad_quant=grad_quantizer, momentum_quant=momentum_quantizer, acc_quant=acc_quantizer)
            if swa_flag==True:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                swa_scheduler = SWALR(optimizer, swa_lr= swa_lr)

                swa_scheduler.load_state_dict(checkpoint['swa_sched'])
                scheduler.load_state_dict(checkpoint['sched'])

    lr = optimizer.param_groups[0]['lr']

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

    

    if swa_flag==True and (epoch + 1) >= swa_start_epoch:
        
        swa_scheduler.step()
    else:
        scheduler.step()

    if swa_flag==True and (epoch + 1) >= swa_start_epoch and (epoch + 1 - swa_start_epoch) % swa_c_epochs == 0:
        

        if swa_model is None:
            print("initialising swa model")
            swa_model = AveragedModel(model)
            # pdb.set_trace()
            print("model weights before swa initialisation: ", model.fc1.weight)
            print("swa_model weights after initialisation: ", swa_model.module.fc1.weight)
        else:
            print("updating swa model")
            swa_model.update_parameters(model)

        print("swa_model weights after update: ", swa_model.module.fc1.weight)

        if epoch == 0 or epoch % 5 == 4 or epoch == (epochs - 1):
            update_bn(input, swa_model, device="cpu")
            print("SWA eval")
            
        else:
            swa_res = {'loss': None, 'accuracy': None}


    epc_num = epoch+1
    if (epc_num == 4 or epc_num == 9 or epc_num == 14):
        torch.save({
            'epoch': epc_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'swa_sched' : swa_scheduler.state_dict(),
            'sched' : scheduler.state_dict(),
            'loss': loss
            }, PATH)




    