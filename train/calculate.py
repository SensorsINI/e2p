"""
 @Time    : 29.05.22 17:13
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : calculate.py
 @Function:
 
"""
import torch
import argparse
import collections
from thop import profile
from thop import clever_format
from parse_config import ConfigParser

# import model.model as module_arch
import model.model_mhy as module_arch

args = argparse.ArgumentParser(description='PyTorch Training')
args.add_argument('-c', '--config', default='./config/m8.json', type=str,
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

model = config.init_obj('arch', module_arch)

input = torch.randn(1, 5, 480, 640)

flops, params = profile(model, inputs=(input,))
flops, params = clever_format([flops, params], "%.3f")
print(model)
print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
