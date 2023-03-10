"""
 @Time    : 02.09.22 10:49
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : model_ed.py
 @Function:
 
"""
import torch
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .submodules import *

from .model_util import *


class BaseUNet(nn.Module):
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks,
                 num_output_channels, skip_type, norm, use_upsample_conv,
                 num_bins, recurrent_block_type=None, kernel_size=5,
                 channel_multiplier=2,
                 ckpt_aolp=None, ckpt_dolp=None):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        # for ec loss
        self.ckpt_aolp = ckpt_aolp
        self.ckpt_dolp = ckpt_dolp

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]
        self.skip_ftn = eval('skip_' + skip_type)
        print('Using skip: {}'.format(self.skip_ftn))
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert(self.num_output_channels > 0)
        print(f'Kernel size {self.kernel_size}')
        print(f'Skip type {self.skip_type}')
        print(f'norm {self.norm}')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                         num_output_channels, 1, activation=None, norm=norm)


class UNetRecurrent(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=2,
                              padding=self.kernel_size // 2)

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return img


def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)


class E2VIDRecurrent(BaseModel):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict


################################################################################################
################################################################################################
################################################################################################
class ED0(BaseModel):
    """
    original e2vid for iad, respectively
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.branch_i = UNetRecurrent(unet_kwargs)
        self.branch_a = UNetRecurrent(unet_kwargs)
        self.branch_d = UNetRecurrent(unet_kwargs)

    # i
    @property
    def states_i(self):
        return copy_states(self.branch_i.states)

    @states_i.setter
    def states_i(self, states):
        self.branch_i.states = states

    def reset_states_i(self):
        self.branch_i.states = [None] * self.branch_i.num_encoders

    # a
    @property
    def states_a(self):
        return copy_states(self.branch_a.states)

    @states_a.setter
    def states_a(self, states):
        self.branch_a.states = states

    def reset_states_a(self):
        self.branch_a.states = [None] * self.branch_a.num_encoders

    # d
    @property
    def states_d(self):
        return copy_states(self.branch_d.states)

    @states_d.setter
    def states_d(self, states):
        self.branch_d.states = states

    def reset_states_d(self):
        self.branch_d.states = [None] * self.branch_d.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: i a d
        """
        i = self.branch_i.forward(event_tensor)
        a = self.branch_a.forward(event_tensor)
        d = self.branch_d.forward(event_tensor)

        return {
            'i': i,
            'a': a,
            'd': d
        }


################################################################################################
######################################## Original + RPPP #######################################
################################################################################################
class UNetRecurrent_RPPP(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        # self.head = ConvLayer(self.num_bins, self.base_num_channels,
        #                       kernel_size=self.kernel_size, stride=2,
        #                       padding=self.kernel_size // 2)
        self.head = RPPP_23(self.num_bins, self.base_num_channels)

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return img


class ED1(BaseModel):
    """
    original e2vid + rppp for iad
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.branch_i = UNetRecurrent_RPPP(unet_kwargs)
        self.branch_a = UNetRecurrent_RPPP(unet_kwargs)
        self.branch_d = UNetRecurrent_RPPP(unet_kwargs)

    # i
    @property
    def states_i(self):
        return copy_states(self.branch_i.states)

    @states_i.setter
    def states_i(self, states):
        self.branch_i.states = states

    def reset_states_i(self):
        self.branch_i.states = [None] * self.branch_i.num_encoders

    # a
    @property
    def states_a(self):
        return copy_states(self.branch_a.states)

    @states_a.setter
    def states_a(self, states):
        self.branch_a.states = states

    def reset_states_a(self):
        self.branch_a.states = [None] * self.branch_a.num_encoders

    # d
    @property
    def states_d(self):
        return copy_states(self.branch_d.states)

    @states_d.setter
    def states_d(self, states):
        self.branch_d.states = states

    def reset_states_d(self):
        self.branch_d.states = [None] * self.branch_d.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: i a d
        """
        i = self.branch_i.forward(event_tensor)
        a = self.branch_a.forward(event_tensor)
        d = self.branch_d.forward(event_tensor)

        return {
            'i': i,
            'a': a,
            'd': d
        }


################################################################################################
############################## Original + RPPP + EC loss #######################################
################################################################################################
class UNetRecurrent_RPPP_ECL(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    Embedding consistency loss.
    """
    def __init__(self, unet_kwargs, flag):
        self.flag = flag
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        self.head = RPPP_23(self.num_bins, self.base_num_channels)

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()

        # same with autoencoder
        norm_layer = nn.BatchNorm2d
        leaky = True
        upsampling = 'bilinear'

        self.Up4 = Up(self.encoder_output_sizes[-1], self.encoder_output_sizes[-2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv4 = RB(self.encoder_output_sizes[-2], self.encoder_output_sizes[-2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.encoder_output_sizes[-2], self.encoder_output_sizes[-3], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv3 = RB(self.encoder_output_sizes[-3], self.encoder_output_sizes[-3], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.encoder_output_sizes[-3], self.encoder_input_sizes[-3], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv2 = RB(self.encoder_input_sizes[-3], self.encoder_input_sizes[-3], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.encoder_input_sizes[-3], 1, kernel_size=1, stride=1, padding=0)

        self.init_decoder(self.flag)

        self.states = [None] * self.num_encoders

    def init_decoder(self, flag):
        if flag == "aolp":
            path = self.ckpt_aolp
            print('AoLP decoder initialized with {}...'.format(self.ckpt_aolp))
        elif flag == "dolp":
            path = self.ckpt_dolp
            print('DoLP decoder initialized with {}...'.format(self.ckpt_dolp))
        else:
            print('Intensity decoder initialized randomly')
            return

        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        print('original keys', state_dict.keys())
        # load autoencoder's decoder weights to ECNet
        state_dict_new = dict({})
        for key in state_dict.keys():
            if 'Up' in key:
                # load weights for decoder
                state_dict_new[key] = state_dict[key]
            if 'Conv.' in key:
                # load weights of last Conv
                state_dict_new[key] = state_dict[key]
        print('new keys', state_dict_new.keys())
        model_dict = self.state_dict()
        model_dict.update(state_dict_new)
        self.load_state_dict(model_dict)

        print("Initialization done!")

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        blocks = []

        # head
        x = self.head(x)

        # encoder
        for i, encoder in enumerate(self.encoders):
            blocks.append(x)
            x, state = encoder(x, self.states[i])
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        e4 = x

        # original decoder
        # for i, decoder in enumerate(self.decoders):
        #     x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # ec loss decoder
        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4)
        d4 = self.skip_ftn(d4, blocks[-1])

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)
        d3 = self.skip_ftn(d3, blocks[-2])

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)
        d2 = self.skip_ftn(d2, blocks[-3])

        out = self.Conv(d2)

        # remove activation
        # if self.final_activation is not None:
        #     img = self.final_activation(out)

        return out, e4


class ED2(BaseModel):
    """
    original e2vid + rppp for iad
    i: p
    a: ec + l2 + ssim
    d: ec + l2 + ssim
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.branch_i = UNetRecurrent_RPPP_ECL(unet_kwargs, flag='intensity')
        self.branch_a = UNetRecurrent_RPPP_ECL(unet_kwargs, flag='aolp')
        self.branch_d = UNetRecurrent_RPPP_ECL(unet_kwargs, flag='dolp')

    # i
    @property
    def states_i(self):
        return copy_states(self.branch_i.states)

    @states_i.setter
    def states_i(self, states):
        self.branch_i.states = states

    def reset_states_i(self):
        self.branch_i.states = [None] * self.branch_i.num_encoders

    # a
    @property
    def states_a(self):
        return copy_states(self.branch_a.states)

    @states_a.setter
    def states_a(self, states):
        self.branch_a.states = states

    def reset_states_a(self):
        self.branch_a.states = [None] * self.branch_a.num_encoders

    # d
    @property
    def states_d(self):
        return copy_states(self.branch_d.states)

    @states_d.setter
    def states_d(self, states):
        self.branch_d.states = states

    def reset_states_d(self):
        self.branch_d.states = [None] * self.branch_d.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: i a d
        """
        i, _ = self.branch_i.forward(event_tensor)
        a, a_embedding = self.branch_a.forward(event_tensor)
        d, d_embedding = self.branch_d.forward(event_tensor)

        return {
            'i': (i + 1) / 2,
            'a': (a + 1) / 2,
            'd': (d + 1) / 2,
            'a_embedding': a_embedding,
            'd_embedding': d_embedding,
        }


###############################################################################################
################## Original + RPPP + EC loss + ESA Skip-Connection ############################
###############################################################################################
class UNetRecurrent_RPPP_ECL_ESASC(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    Embedding consistency loss.
    """
    def __init__(self, unet_kwargs, flag):
        self.flag = flag
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        self.head = RPPP_23(self.num_bins, self.base_num_channels)

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()

        self.esasc2 = ESA(self.encoder_input_sizes[0], self.encoder_input_sizes[0], nn.Conv2d)
        self.esasc3 = ESA(self.encoder_input_sizes[1], self.encoder_input_sizes[1], nn.Conv2d)
        self.esasc4 = ESA(self.encoder_input_sizes[2], self.encoder_input_sizes[2], nn.Conv2d)

        # same with autoencoder
        norm_layer = nn.BatchNorm2d
        leaky = True
        upsampling = 'bilinear'

        self.Up4 = Up(self.encoder_output_sizes[-1], self.encoder_output_sizes[-2], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv4 = RB(self.encoder_output_sizes[-2], self.encoder_output_sizes[-2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.encoder_output_sizes[-2], self.encoder_output_sizes[-3], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv3 = RB(self.encoder_output_sizes[-3], self.encoder_output_sizes[-3], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.encoder_output_sizes[-3], self.encoder_input_sizes[-3], norm_layer, leaky=leaky, bilinear=upsampling == 'bilinear')
        self.Up_conv2 = RB(self.encoder_input_sizes[-3], self.encoder_input_sizes[-3], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.encoder_input_sizes[-3], 1, kernel_size=1, stride=1, padding=0)

        self.init_decoder(self.flag)

        self.states = [None] * self.num_encoders

    def init_decoder(self, flag):
        if flag == "aolp":
            path = self.ckpt_aolp
            print('AoLP decoder initialized with {}...'.format(self.ckpt_aolp))
        elif flag == "dolp":
            path = self.ckpt_dolp
            print('DoLP decoder initialized with {}...'.format(self.ckpt_dolp))
        else:
            print('Intensity decoder initialized randomly')
            return

        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        print('original keys', state_dict.keys())
        # load autoencoder's decoder weights to ECNet
        state_dict_new = dict({})
        for key in state_dict.keys():
            if 'Up' in key:
                # load weights for decoder
                state_dict_new[key] = state_dict[key]
            if 'Conv.' in key:
                # load weights of last Conv
                state_dict_new[key] = state_dict[key]
        print('new keys', state_dict_new.keys())
        model_dict = self.state_dict()
        model_dict.update(state_dict_new)
        self.load_state_dict(model_dict)

        print("Initialization done!")

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        blocks = []

        # head
        x = self.head(x)

        # encoder
        for i, encoder in enumerate(self.encoders):
            blocks.append(x)
            x, state = encoder(x, self.states[i])
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        e4 = x

        # original decoder
        # for i, decoder in enumerate(self.decoders):
        #     x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # ec loss decoder
        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4)
        d4 = self.skip_ftn(d4, self.esasc4(blocks[-1]))

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)
        d3 = self.skip_ftn(d3, self.esasc3(blocks[-2]))

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)
        d2 = self.skip_ftn(d2, self.esasc2(blocks[-3]))

        out = self.Conv(d2)

        # remove activation
        # if self.final_activation is not None:
        #     img = self.final_activation(out)

        return out, e4


class ED3(BaseModel):
    """
    original e2vid + rppp for iad
    i: p
    a: ec + l2 + ssim
    d: ec + l2 + ssim
    esa for skip-connection
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.branch_i = UNetRecurrent_RPPP_ECL_ESASC(unet_kwargs, flag='intensity')
        self.branch_a = UNetRecurrent_RPPP_ECL_ESASC(unet_kwargs, flag='aolp')
        self.branch_d = UNetRecurrent_RPPP_ECL_ESASC(unet_kwargs, flag='dolp')

    # i
    @property
    def states_i(self):
        return copy_states(self.branch_i.states)

    @states_i.setter
    def states_i(self, states):
        self.branch_i.states = states

    def reset_states_i(self):
        self.branch_i.states = [None] * self.branch_i.num_encoders

    # a
    @property
    def states_a(self):
        return copy_states(self.branch_a.states)

    @states_a.setter
    def states_a(self, states):
        self.branch_a.states = states

    def reset_states_a(self):
        self.branch_a.states = [None] * self.branch_a.num_encoders

    # d
    @property
    def states_d(self):
        return copy_states(self.branch_d.states)

    @states_d.setter
    def states_d(self, states):
        self.branch_d.states = states

    def reset_states_d(self):
        self.branch_d.states = [None] * self.branch_d.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: i a d
        """
        i, _ = self.branch_i.forward(event_tensor)
        a, a_embedding = self.branch_a.forward(event_tensor)
        d, d_embedding = self.branch_d.forward(event_tensor)

        return {
            'i': (i + 1) / 2,
            'a': (a + 1) / 2,
            'd': (d + 1) / 2,
            'a_embedding': a_embedding,
            'd_embedding': d_embedding,
        }
