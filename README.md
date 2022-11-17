# Training with Block Minifloat in Pytorch

This repository is based on the code to accompany the paper [A Block Minifloat Representation for Training Deep Neural Networks](https://openreview.net/forum?id=6zaTwpNSsQ2).

This project builds up on the work done by Fox et al. in https://github.com/sfox14/block_minifloat .

Some example commands have been provided to train the models using the various implementations.

## Requirements
python>=3.6
pytorch>=1.1


## Usage
```bash
python main.py --data_path=. --dataset=CIFAR10 --model=VGG16LP --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=10 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=4 --error-man=3
```

This will train a ResNet-18 model using the BM8 format specified in the paper.


Train a model on the CIFAR10 dataset.
```bash
python3 main.py --data_path=. --dataset=CIFAR10 --model=VGG16LP --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=15 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=2 --error-man=5 --adaptive_scale=False --k=5
```


To train a model with adaptive scaling, use the following command:

```bash
python main.py --data_path=. --dataset=CIFAR10 --model=ResNet18LP --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=10 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=4 --error-man=3 --adaptive_scale=True --adaptive_start=2 --k=3
```


To train a model with adaptive scaling and Stochastic Weight Averaging (SWA) optimisation, use the following command:

```bash
python main.py --data_path=. --dataset=CIFAR10 --model=ResNet18LP --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=10 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=4 --error-man=3 --adaptive_scale=True --adaptive_start=2 --k=3 --swa --swa_start=5 --swa_lr=0.01
```

Train a model using Cyclic SBM

```bash
python3 main.py --data_path=. --dataset=MNIST --model=MNet --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=30 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=2 --error-man=5 --adaptive_scale=False --k=1 --cpt=True --num_cyclic_period=6 --cyclic_fwd_k_schedule 1 4
```

Finally, to train a few models multiple times to replicate experimentation results, change the content of run_experiment.sh and run.
