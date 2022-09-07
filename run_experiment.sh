#!/bin/bash
### use MobileNetV2LP for model
model=$1
epochs=30
adaptive_scale=False

for k in {1..5}
do
    python3 main.py --data_path=. --dataset=CIFAR100 --model=$model --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=$epochs \
        --weight-exp=2 --weight-man=5 \
        --activate-exp=2 --activate-man=5 \
        --error-exp=2 --error-man=5 --adaptive_scale=$adaptive_scale --k=$k | tee "${model}_25_adaptive_scale_${adaptive_scale}_k_${k}.out"
    
done

adaptive_scale='True'
k_array=(5 3)

for k_val in ${k_array[@]}
do
    python3 main.py --data_path=. --dataset=CIFAR100 --model=$model --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=$epochs \
        --weight-exp=2 --weight-man=5 \
        --activate-exp=2 --activate-man=5 \
        --error-exp=2 --error-man=5 --adaptive_scale=$adaptive_scale --k=${k_val} > "${model}_25_adaptive_scale_${adaptive_scale}_k_${k_val}.out"
    
done