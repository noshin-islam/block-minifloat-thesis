"""
Code modified from Qpytorch repository. https://github.com/Tiiiger/QPyTorch/blob/master
"""

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

from torch.optim.swa_utils import AveragedModel, update_bn, SWALR

# import PrettyTable
#from qtorch import FloatingPoint

if __name__ == "__main__":

    num_types = ["weight", "activate", "error", "acc", "grad", "momentum"]

    parser = argparse.ArgumentParser(description='Block Minifloat SGD training')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100','IMAGENET', 'MNIST', 'TINY_IMAGENET'],
                        help='dataset name: CIFAR10, CIFAR100, TINY_IMAGENET or IMAGENET')
    parser.add_argument('--data_path', type=str, default="/opt/datasets/", required=True, metavar='PATH',
                        help='path to datasets location (default: "./data")')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--model', type=str, default="ResNet18LP", required=True, metavar='MODEL',
                        help='model name (default: ResNet18LP)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                        help='evaluation frequency (default: 5)')
    parser.add_argument('--save_freq', type=int, default=15, metavar='N',
                        help='save checkpoint frequency (default: 5)')
    parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=200, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--tile', type=int, default=48, choices=[16, 24, 32, 48, -1, 0],
                        help='tile size for shared exponent (default: -1); -1 if image tensor')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--flush_to_zero', action='store_true', default=False,
                        help='use qtorch floating point quantizer which does not handle denormal numbers')
    parser.add_argument('--gpu', type=str, help='comma separated list of GPU(s) to use')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='output directory')

    #take scaling factor as cmd line arg
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--adaptive_scale', type=str, default='False')
    parser.add_argument('--adaptive_start', type=int, default=5)

    # SWA implementation: https://github.com/izmailovpavel/torch_swa_examples/blob/master/train.py

    parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
    parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
    parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
    parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                        help='SWA model collection frequency/cycle length in epochs (default: 1)')
    parser.add_argument('--swa_on_cpu', action='store_true', help='store swa model on cpu flag (default: off)')

    # adding cpt
    parser.add_argument('--cpt', type=str, default='False')
    parser.add_argument('--num_cyclic_period', type=int, default=1)
    parser.add_argument('--cyclic_fwd_k_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision using k')
    parser.add_argument('--cyclic_bw_k_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision using k')
    

    for num in num_types:
        parser.add_argument('--{}-man'.format(num), type=int, default=-1, metavar='N',
                            help='number of bits to use for mantissa of {}; -1 if full precision.'.format(num))
        parser.add_argument('--{}-exp'.format(num), type=int, default=-1, metavar='N',
                            help='number of bits to use for exponent of {}; -1 if full precision.'.format(num))
        parser.add_argument('--{}-rounding'.format(num), type=str, default='stochastic', metavar='S',
                            choices=["stochastic","nearest"],
                            help='rounding method for {}, stochastic or nearest'.format(num))

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    utils.set_seed(args.seed, args.cuda)

    ##edit: to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # gpu to use
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Setup log directory
    time_str = time.strftime("%m_%d_%H_%M")
    iden = "{}_{}_w{}{}_a{}{}_e{}{}".format(args.model, args.dataset, 
        args.weight_man, args.weight_exp, 
        args.activate_man, args.activate_exp,
        args.error_man, args.error_exp)
    dir_name = os.path.join(args.output_dir, iden)

    log_name = os.path.join(dir_name, "main_{}.log".format(time_str))
    print('Checkpoint directory {}'.format(dir_name))
    os.makedirs(dir_name, exist_ok=True)
    with open(log_name, 'w') as f: f.write('python {}\n\n'.format(sys.argv))

    # load dataset
    loaders = get_data(args.dataset, args.data_path, args.batch_size)

    # define quantizers
    number_dict = {}
    for i, num in enumerate(num_types):
        num_rounding = getattr(args, "{}_rounding".format(num))
        num_man = getattr(args, "{}_man".format(num))
        num_exp = getattr(args, "{}_exp".format(num))
        number_dict[num] = BlockMinifloat(exp=num_exp, man=num_man, 
                tile=args.tile, flush_to_zero=args.flush_to_zero, k_exp= args.k)
        print("{:10}: {}".format(num, number_dict[num]))

    weight_quantizer = quantizer(forward_number=number_dict["weight"],
                                forward_rounding=args.weight_rounding)
    grad_quantizer   = quantizer(forward_number=number_dict["grad"],
                                forward_rounding=args.grad_rounding)
    momentum_quantizer = quantizer(forward_number=number_dict["momentum"],
                                forward_rounding=args.momentum_rounding)
    acc_quantizer = quantizer(forward_number=number_dict["acc"],
                                forward_rounding=args.acc_rounding)
    acc_err_quant = lambda : Quantizer(number_dict["activate"], number_dict["error"],
                                        args.activate_rounding, args.error_rounding)

    # Build model
    seed = 3
    torch.manual_seed(seed)

    print('Model: {}'.format(args.model))
    if args.adaptive_scale == 'True':
        print(f"Adaptive Scaling being used, starting from k = {args.k}")
    else:
        print(f"Standard exponent scaling of k = {args.k} being used")

    model_cfg = getattr(models, args.model)

    model_cfg.kwargs.update({"quant":acc_err_quant})

    if args.dataset=="CIFAR10": num_classes=10
    elif args.dataset=="CIFAR100": num_classes=100
    elif args.dataset=="IMAGENET": num_classes=1000
    elif args.dataset=="MNIST": num_classes=10
    elif args.dataset=="TINY_IMAGENET": num_classes=200


    if (args.model == "ResNet18LP") or (args.model == "ResNet50LP") or (args.model == "MobileNetV2LP"): 
        model_cfg.kwargs.update({"image_size":224 if (args.dataset=="IMAGENET") else 64 if (args.dataset=="TINY_IMAGENET") else 28 if args.dataset=="MNIST" else 32})
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

    # for name, param in model.named_parameters():
    #     print(f"name: {name}")
    #     print(f"value: {param}")



    def optimizer_to(optim, device):
    # move optimizer to device
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)



    def change_scale(model, optimizer, k_val, weight_bm, activate_bm, error_bm, acc_bm, grad_bm, momentum_bm):
        print("k value switch")
        
        print(f'Old k = {weight_bm.k_exp}')

        weight_bm.change_k(k_val)
        activate_bm.change_k(k_val)

        #uncomment the commented out lines to enable Cyclic Scaling in both forward and backward direction

        # error_bm.change_k(k_val)
        # acc_bm.change_k(k_val)     #accuracy
        # grad_bm.change_k(k_val)
        # momentum_bm.change_k(k_val)

        print(f'New k = {weight_bm.k_exp}')

        weight_quantizer = quantizer(forward_number=weight_bm, forward_rounding='stochastic')

        # grad_quantizer   = quantizer(forward_number=grad_bm, forward_rounding='stochastic')

        # momentum_quantizer = quantizer(forward_number=momentum_bm, forward_rounding='stochastic')

        # acc_quantizer = quantizer(forward_number=acc_bm, forward_rounding='stochastic')

        acc_err_quant = lambda : Quantizer(activate_bm, error_bm, 'stochastic', 'stochastic')

        # model.quant = acc_err_quant()
        model.modify_layer_quant(acc_err_quant)
        # print("model weights after loading checkpoint: ", model.fc1.weight)

        # updating the optimiser
        # print("updating the quantisers in optimizer")
        # optimizer.change_quantizers(weight_quantizer, grad_quantizer, momentum_quantizer, acc_quantizer) #use this line for cyclic scaling on both directions
        optimizer.change_fwd_quantizer(weight_quantizer) #comment this out if you've uncommented above line of code -- this one is only when using cyclic scaling in forward dir

        return model, optimizer


    def cyclic_adjust_precision(args, _iter, cyclic_period):
        assert len(args.cyclic_fwd_k_schedule) == 2
        fwd_k_min = args.cyclic_fwd_k_schedule[0]
        fwd_k_max = args.cyclic_fwd_k_schedule[1]


        forward_k = np.rint(fwd_k_min +
                                0.5 * (fwd_k_max - fwd_k_min) *
                                (1 - np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        
        return forward_k

            
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
        
    # count_parameters(model)

    ####
    model.to(device)
    

    ## SWA initialisation
    if args.swa:
        print('SWA training')
        swa_model = None
    else:
        print('SGD training')

    # learning rate schedules
    def default_schedule(epoch):
        t = (epoch) / (args.swa_start if args.swa else args.epochs)
        lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
        t_const = 0.5 # 0.2 previous
        if t <= t_const:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - t_const) / (0.9 - t_const)
        else:
            factor = lr_ratio
        return args.lr_init * factor

    # https://github.com/uoguelph-mlrg/Cutout (200 epochs)
    def cifar_schedule(epoch): 
        milestones = [60, 120, 160]
        gamma = 0.2
        lr_init = 0.05
        return lr_init * (gamma ** bisect.bisect_right(milestones, epoch) ) 

    # ptorch examples (90 epochs)
    def imagenet_schedule(epoch):
        lr_init = args.lr_init
        lr = lr_init * (0.1 ** (epoch // 30))
        return lr


    if (args.dataset == "CIFAR10" or args.dataset == "CIFAR100") and args.epochs >= 200:
        schedule = lambda x: cifar_schedule(x)
    elif args.dataset == "IMAGENET" or args.dataset == "TINY_IMAGENET":
        schedule = lambda x: imagenet_schedule(x)
    else:
        schedule = lambda x: default_schedule(x)


    criterion = F.cross_entropy
    optimizer = SGD( model.parameters(),
                    lr=args.lr_init,
                    momentum=0.9,
                    weight_decay=args.wd)


    # resume training
    start_epoch = 0
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        resume_keys = list(checkpoint['state_dict'])
        model_keys = list(model.state_dict())
        matched_state_dict = {
            model_keys[i]:checkpoint['state_dict'][k] for i,k in enumerate(resume_keys)}
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(matched_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.swa:
            swa_state_dict = checkpoint['swa_state_dict']
            if swa_state_dict is not None:
                swa_model.load_state_dict(swa_state_dict)


    optimizer = OptimLP(optimizer,
                        weight_quant=weight_quantizer,
                        grad_quant=grad_quantizer,
                        momentum_quant=momentum_quantizer,
                        acc_quant=acc_quantizer)

    
    ## defining the SWA schedulers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    
    # Prepare logging
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'te_time']
    if args.swa:
        columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
        print(f"Columns look like: {columns}")
        swa_res = {'loss': None, 'accuracy': None}

    # k_value = 3
    checkpoint_num = 5
    PATH = 'testlogs/checkpoint.pth'

    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        if not args.swa:
            lr = schedule(epoch)
            utils.adjust_learning_rate(optimizer, lr)
        else:
            lr = optimizer.param_groups[0]['lr']

        if args.cpt == 'True':
            cyclic_period = int(args.epochs / args.num_cyclic_period)
            k_new = cyclic_adjust_precision(args, epoch, cyclic_period)
            model, optimizer = change_scale(model, optimizer, k_new, number_dict["weight"], number_dict["activate"], number_dict["error"], number_dict["acc"], number_dict["grad"], number_dict["momentum"])

    
        # ADAPTIVE SCALING SECTION
        if args.adaptive_scale == 'True':
            
            if ((epoch+1) != 1 and number_dict["weight"].k_exp != 1 and (epoch+1) % args.adaptive_start == 0):
                print("K value switch")
                # weight activate error accuracy grad momentum
                print(f'Old k = {number_dict["weight"].k_exp}')

                #updating the k values by subtracting 1
                number_dict["weight"].change_k(number_dict["weight"].k_exp-1)
                number_dict["activate"].change_k(number_dict["activate"].k_exp-1)
                number_dict["error"].change_k(number_dict["error"].k_exp-1)
                number_dict["acc"].change_k(number_dict["acc"].k_exp-1) #accuracy
                number_dict["grad"].change_k(number_dict["grad"].k_exp-1)
                number_dict["momentum"].change_k(number_dict["momentum"].k_exp-1)

                print(f'New k = {number_dict["weight"].k_exp}')

                weight_quantizer = quantizer(forward_number=number_dict["weight"], forward_rounding=args.weight_rounding)
                grad_quantizer   = quantizer(forward_number=number_dict["grad"], forward_rounding=args.grad_rounding)
                momentum_quantizer = quantizer(forward_number=number_dict["momentum"], forward_rounding=args.momentum_rounding)
                acc_quantizer = quantizer(forward_number=number_dict["acc"], forward_rounding=args.acc_rounding)
                acc_err_quant = lambda : Quantizer(number_dict["activate"], number_dict["error"], args.activate_rounding, args.error_rounding)

                model_cfg = getattr(models, args.model)
                model_cfg.kwargs.update({"quant":acc_err_quant})
                model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
                # print("model weights before loading checkpoint: ", model.fc1.weight)

                ## loading back the model state dictionary into the newly defined model
                checkpoint = torch.load(PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                # print("model weights after loading checkpoint: ", model.fc1.weight)

                # loading in the new optimiser
                print("loading checkpoint - old model, opt, sched")
                criterion = F.cross_entropy
                optimizer = SGD(model.parameters(), lr = args.lr_init, momentum=0.9, weight_decay = args.wd)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer = OptimLP(optimizer, weight_quant=weight_quantizer, grad_quant=grad_quantizer, momentum_quant=momentum_quantizer, acc_quant=acc_quantizer)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

                
                scheduler.load_state_dict(checkpoint['scheduler'])
                swa_scheduler.load_state_dict(checkpoint['swa_scheduler'])
                # epoch = checkpoint['epoch']
                # loss = checkpoint['loss']

                


        train_res = utils.run_epoch(loaders['train'], model, criterion,
                                    optimizer=optimizer, phase="train" )
        time_pass = time.time() - time_ep
        train_res['time_pass'] = time_pass

        ## saving the model and optimiser state for adaptive scaling
        epc_num = epoch+1
        if (args.adaptive_scale == 'True' and epc_num % args.adaptive_start == (args.adaptive_start-1) and number_dict["weight"].k_exp != 1):
            print("Saving checkpoint for adaptive scaling...")
            torch.save({
                'epoch': epc_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'swa_scheduler': swa_scheduler.state_dict(),
                'loss': train_res['loss']
                }, PATH)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            time_ep = time.time()
            
            utils.bn_update(loaders['train'], model)
            test_res = utils.run_epoch(loaders['test'], model, criterion, phase="eval") 

            time_pass = time.time() - time_ep
            test_res['time_pass'] = time_pass
        else:
            test_res = {'loss': None, 'accuracy': None, 'time_pass': None}

        ## SWA to be used for model in here

        # lr = optimizer.param_groups[0]['lr']

        if args.swa and (epoch + 1) >= args.swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()

        if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
            if swa_model is None:
                # print("initialising swa model")
                swa_model = AveragedModel(model)
            else:
                # print("updating swa model")
                swa_model.update_parameters(model)

            if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
                update_bn(loaders['train'], swa_model, device=torch.device('cuda'))
                if args.swa_on_cpu:
                    # moving swa_model to gpu for evaluation
                    model = model.cpu()
                    swa_model = swa_model.to(device)
                print("SWA eval")
                swa_res = utils.run_epoch(loaders['test'], swa_model, criterion, phase="eval")
                if args.swa_on_cpu:
                    model = model.to(device)
                    swa_model = swa_model.cpu()
            else:
                swa_res = {'loss': None, 'accuracy': None}

        ## END OF NEW SWA SECTION
        if args.swa:
            values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], train_res['time_pass'], 
                    test_res['loss'], test_res['accuracy'], swa_res['loss'], swa_res['accuracy'], test_res['time_pass']]
        else:
            values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], train_res['time_pass'], 
                    test_res['loss'], test_res['accuracy'], test_res['time_pass']]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        with open(log_name, 'a') as f: f.write(table+'\n')

        # save checkpoint by epoch
        if epoch % args.save_freq == args.save_freq - 1 or epoch == args.epochs - 1:
            base_dir = os.path.join(dir_name, "base")
            os.makedirs(base_dir, exist_ok=True)
            utils.save_checkpoint(
                base_dir,
                epoch+1,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )

        # save the most recent checpoint every epoch
        utils.save_checkpoint(
            dir_name,
            epoch+1,
            recent=True,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )



