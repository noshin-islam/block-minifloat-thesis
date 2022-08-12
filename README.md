# Training with Block Minifloat in Pytorch

This repository provides code to accompany the paper [A Block Minifloat Representation for Training Deep Neural Networks](https://openreview.net/forum?id=6zaTwpNSsQ2).

This project builds up on the work done by Fox et al. in https://github.com/sfox14/block_minifloat .

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

python3 main.py --data_path=. --dataset=MNIST --model=MNet --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=15 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=2 --error-man=5 --adaptive_scale=False --k=5
