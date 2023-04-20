#!/bin/bash

# Train a model from scratch
export CUDA_VISIBLE_DEVICES=0 # set chosen GPU if there is more than one. Use nvidia-smi -L to list them
pdavis='/home/tobi/pdavis_demo'
#python my_train.py --config ./config_v/v16_b_c16_i.json --limited_memory
#python my_train.py --config ./config_v/v16_so.json --limited_memory
#python my_train.py --config ./e2p.json --limited_memory # limited_memory reduces open file memory use

# Train model from specific checkpoint
#export CUDA_VISIBLE_DEVICES=0
#python my_train.py --config ./config_v/v16_woln_ft_mix.json --limited_memory --resume /home/mhy/firenet-pdavis/ckpt/models/v16_woln/1018_202813/model_best.pth

# # fine tune published model with new data; see README.md to add new data
# config="e2p.json"
# echo "training with config $config"
# python my_train.py -c $config --limited_memory --resume $pdavis/e2p-cvpr2023.pth


# fine tune published model with new data; see README.md to add new data
config="e2p-finetune-only-new.json"
ckpt="$pdavis/e2p.pth"
echo "training with config $config starting from ckpt $ckpt"
python train.py -c $config --limited_memory --resume $ckpt
