#!/bin/bash
#BSUB -J inference
#BSUB -q gpu_v100
#BSUB -o %inference.out
#BSUB -e %inference.err
#BSUB -gpu "num=1:mode=exclusive_process"
module load anaconda3
source activate xiaofeng11
#python inference_ICDehazing.py --results_dir ../results/ICDehazing/4KDehazing_L2_DCP_val --img_w 256 --img_h 256 --pth_path ../results/ICDehazing/4KDehazing_L2_DCP/models/last_x2y.pth --dataset 4KDehazing --if_mul_alpha False

# infer different alpha
python inference_ICDehazing.py --results_dir ../results/ICDehazing/OTS_differ_alpha_val --img_w 256 --img_h 256 --pth_path ../results/ICDehazing/OTS_L2_DCP/models/last_x2y.pth --dataset OTS --if_mul_alpha True