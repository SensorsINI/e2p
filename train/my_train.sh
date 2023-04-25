#!/bin/bash

# Train a model from scratch
export CUDA_VISIBLE_DEVICES=0 # set chosen GPU if there is more than one. Use nvidia-smi -L to list them
pdavis='/home/tobi/pdavis_demo'
export PYTHONPATH=$pdavis # necessary to locate e.g. train.xxx packages and scripts

#python my_train.py --config ./config_v/v16_b_c16_i.json --limited_memory
#python my_train.py --config ./config_v/v16_so.json --limited_memory
#python my_train.py --config ./e2p.json --limited_memory # limited_memory reduces open file memory use

# Train model from specific checkpoint
#export CUDA_VISIBLE_DEVICES=0
#python my_train.py --config ./config_v/v16_woln_ft_mix.json --limited_memory --resume /home/mhy/firenet-pdavis/ckpt/models/v16_woln/1018_202813/model_best.pth

# fine tune published model with new data; see README.md to add new data
# config="e2p.json"
# ckpt="$pdavis/e2p.pth"
# echo "training with config $config starting from ckpt $ckpt"
# python my_train.py -c $config --limited_memory --resume $pdavis/e2p-cvpr2023.pth


#  fine tune published model with ONLY new data; see README.md to add new data
# config="e2p-finetune.json"
# ckpt="$pdavis/e2p.pth"
# echo "training with config $config starting from ckpt $ckpt"
# python my_train.py -c $config --limited_memory --resume $ckpt

config="e2p-finetune.json"
ckpt="$pdavis/train/ckpt/models/e2p/0425_121200/model_best.pth"
echo "training with config $config starting from ckpt $ckpt"
nice python my_train.py -c $config --limited_memory --resume $ckpt
