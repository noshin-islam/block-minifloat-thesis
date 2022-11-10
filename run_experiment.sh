#!/bin/bash

model=ResNet18LP
epochs=30

echo "ResNet18LP NUM CYCLING PERIOD = 3"
for i in {1..3}
do
    python3 main.py --dataset=TINY_IMAGENET --data_path='./tiny-imagenet-200' --model=$model --batch_size=256 --wd=5e-4 --lr_init=0.1 --epochs=30 \
      --weight-exp=2 --weight-man=5 \
      --activate-exp=2 --activate-man=5 \
      --error-exp=2 --error-man=5 --k=1 --cpt=True --num_cyclic_period=3 --cyclic_fwd_k_schedule 1 3 | tee "img_5_cycle_cpt_${model}_${i}.out"

done

echo "ResNet18LP NUM CYCLING PERIOD = 5"
for i in {1..3}
do
    python3 main.py --dataset=TINY_IMAGENET --data_path='./tiny-imagenet-200' --model=$model --batch_size=256 --wd=5e-4 --lr_init=0.1 --epochs=30 \
      --weight-exp=2 --weight-man=5 \
      --activate-exp=2 --activate-man=5 \
      --error-exp=2 --error-man=5 --k=1 --cpt=True --num_cyclic_period=5 --cyclic_fwd_k_schedule 1 3 | tee "img_5_cycle_cpt_${model}_${i}.out"

done

echo "ResNet18LP NO CPT"
for i in {1..3}
do
    python3 main.py --dataset=TINY_IMAGENET --data_path='./tiny-imagenet-200' --model=$model --batch_size=256 --wd=5e-4 --lr_init=0.1 --epochs=30 \
      --weight-exp=2 --weight-man=5 \
      --activate-exp=2 --activate-man=5 \
      --error-exp=2 --error-man=5 --k=1 | tee "img_no_cpt_${model}_${i}.out"

done

