import torch
from globals_and_utils import *
log = get_logger(__name__)
from train.model import model_v as model_arch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_info={}
def load_model_from_args(args):
    """ Loads a model given the program arguments
    :param args: argparse returned args, args.checkpoint_path is the full path to the file
    :returns: the pytorch model that has been loaded.
    """
    checkpoint = torch.load(args.checkpoint_path)
    config = checkpoint['config']
    log.info(f"config['arch']={config['arch']}")

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    log.info(f"model_info['num_bins']={model_info['num_bins']}")
    # logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    log.info(f"model={model}")
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    log.info('Load my trained weights succeeded!')

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
