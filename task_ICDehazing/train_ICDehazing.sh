#!/bin/bash
#BSUB -J 4k_DCP
#BSUB -q gpu_v100
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu "num=1:mode=exclusive_process"
module load anaconda3
source activate xiaofeng11
python train_ICDehazing.py --results_dir ../results/ICDehazing/4KDehazing_L2_DCP --img_w 256 --img_h 256 --train_batch_size 2 --dataset 4KDehazing --rec_loss L2 --prior_per True --prior_per_weight 1 --prior_decay 0.9 --model ICDehazing