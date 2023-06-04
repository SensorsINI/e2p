"""
 @Time    : 29.03.22 15:06
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : my_train.py
 @Function:
 
"""
import datetime
import time
import argparse
import collections
import torch
import numpy as np
import train.data_loader.data_loaders as module_data
import train.model.loss as module_loss
# import model.model as module_arch
# import model.model_mhy as module_arch
import train.model.model_v as module_arch
# import model.model_ed as module_arch
from train.parse_config import ConfigParser
from train.trainer import Trainer, Trainer_P, Trainer_S, Trainer_I, Trainer_RP

# fix random seeds for reproducibility
SEED = 2022
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


# def load_model(args, checkpoint=None, config=None):
#     """
#     negative voxel indicates a model trained on negative voxels -.-
#     """
#     resume = checkpoint is not None
#     if resume:
#         config = checkpoint['config']
#         state_dict = checkpoint['state_dict']
#     try:
#         model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
#     except KeyError:
#         model_info['num_bins'] = config['arch']['args']['num_bins']
#     logger = config.get_logger('test')
#
#     if args.legacy:
#         config['arch']['type'] += '_legacy'
#     # build model architecture
#     model = config.init_obj('arch', module_arch)
#     logger.info(model)
#
#     if config['n_gpu'] > 1:
#         model = torch.nn.DataParallel(model)
#     if resume:
#         model.load_state_dict(state_dict)
#
#     # prepare model for testing
#     model = model.to(device)
#     model.eval()
#     if args.color:
#         model = ColorNet(model)
#         print('Loaded ColorNet')
#     return model

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # init loss classes
    loss_ftns = [getattr(module_loss, loss)(**kwargs) for loss, kwargs in config['loss_ftns'].items()]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # trainer = Trainer(model, loss_ftns, optimizer,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    # trainer = Trainer_I(model, loss_ftns, optimizer,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    # trainer = Trainer_S(model, loss_ftns, optimizer,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    # check pytorch version, if >= 2.0 use compile
    version = torch.__version__
    major_version = int(version.split('.')[0])

    if major_version >= 2:
        model = torch.compile(model)
    else:
        pass

    trainer = Trainer_P(model, loss_ftns, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    # trainer = Trainer_RP(model, loss_ftns, optimizer,
    #                     config=config,
    #                     data_loader=data_loader,
    #                     valid_data_loader=valid_data_loader,
    #                     lr_scheduler=lr_scheduler)

    start_time = time.time()
    torch.cuda.empty_cache()
    trainer.train()
    print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    print("Optimization Have Done!")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Training')
    args.add_argument('-c', '--config', default='train/e2p.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--limited_memory', default=False, action='store_true',
                      help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--rmb', '--reset_monitor_best'], type=bool, target='trainer;reset_monitor_best'),
        CustomArgs(['--vo', '--valid_only'], type=bool, target='trainer;valid_only')
    ]
    config = ConfigParser.from_args(args, options)

    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    main(config)

