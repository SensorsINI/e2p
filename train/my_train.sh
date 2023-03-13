# Train a model from scratch
export CUDA_VISIBLE_DEVICES=0
#python train.py --config ./config_v/v16_b_c16_i.json --limited_memory
#python train.py --config ./config_v/v16_so.json --limited_memory
python train.py --config ./e2p.json --limited_memory

# Train s model from specific checkpoint
#export CUDA_VISIBLE_DEVICES=0
#python train.py --config ./config_v/v16_woln_ft_mix.json --limited_memory --resume /home/mhy/firenet-pdavis/ckpt/models/v16_woln/1018_202813/model_best.pth