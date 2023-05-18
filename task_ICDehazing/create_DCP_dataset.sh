#!/bin/bash
#BSUB -J create_dataset
#BSUB -q gpu_v100
#BSUB -o %J.out
#BSUB -e %J.err
module load anaconda3
python create_DCP_dataset.py