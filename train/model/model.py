import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .unet import UNetFlow, WNet, UNetFlowNoRecur, UNetRecurrent, UNet
from .submodules import *
from ..utils.color_utils import merge_channels_into_color_image

from .legacy import *

################# Start M 1 ############
class UNetFire_P(BaseUNet):
    """
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convgru', base_num_channels=32,
                 num_residual_blocks=2, norm=None, kernel_size=3,
                 recurrent_blocks={'resblock': [0]}, BN_momentum=0.1):
        super(UNetFire_P, self).__init__(num_input_channels=num_input_channels,
                                       num_output_channels=num_output_channels,
                                       skip_type=skip_type,
                                       base_num_channels=base_num_channels,
                                       num_residual_blocks=num_residual_blocks,
                                       norm=norm,
                                       kernel_size=kernel_size)

        self.recurrent_blocks = recurrent_blocks
        self.num_recurrent_units = 0
        self.head = RecurrentConvLayer(self.num_input_channels,
                                       self.base_num_channels,
                                       kernel_size=self.kernel_size,
                                       padding=self.kernel_size // 2,
                                       recurrent_block_type=recurrent_block_type,
                                       norm=self.norm,
                                       BN_momentum=BN_momentum)
        self.num_recurrent_units += 1
        self.resblocks = nn.ModuleList()
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i in range(self.num_residual_blocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                self.resblocks.append(RecurrentResidualLayer(
                    in_channels=self.base_num_channels,
                    out_channels=self.base_num_channels,
                    recurrent_block_type=recurrent_block_type,
                    norm=self.norm,
                    BN_momentum=BN_momentum))
                self.num_recurrent_units += 1
            else:
                self.resblocks.append(ResidualBlock(self.base_num_channels,
                                                    self.base_num_channels,
                                                    norm=self.norm,
                                                    BN_momentum=BN_momentum))
        # intensity
        self.pred_i = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                              self.num_output_channels, kernel_size=1, stride=2, padding=0, activation=None, norm=None)
        self.pred_i.conv2d.bias.data.fill_(0.5)
        # aolp
        self.pred_a = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                                self.num_output_channels, kernel_size=1, stride=2, padding=0, activation=None,
                                norm=None)
        self.pred_a.conv2d.bias.data.fill_(0.5)
        # dolp
        self.pred_d = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                                self.num_output_channels, kernel_size=1, stride=2, padding=0, activation=None,
                                norm=None)
        self.pred_d.conv2d.bias.data.fill_(0.5)

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H/2 x W/2,
                 N x num_output_channels x H/2 x W/2,
                 N x num_output_channels x H/2 x W/2
        """

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        # head
        x, state = self.head(x, prev_states[state_idx])
        state_idx += 1
        states.append(state)

        head_feature_map = x

        # residual blocks
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i, resblock in enumerate(self.resblocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                x, state = resblock(x, prev_states[state_idx])
                state_idx += 1
                states.append(state)
            else:
                x = resblock(x)

        # tail
        y = self.apply_skip_connection(x, head_feature_map)

        intensity = self.pred_i(y)
        aolp = self.pred_a(y)
        dolp = self.pred_d(y)

        return intensity, aolp, dolp, states


class M1(BaseE2VID):
    """
    Use FireNet_legacy to estimate polarization directly.
    """
    def __init__(self, config={}, unet_kwargs={}):
        if unet_kwargs:
            config = unet_kwargs
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        BN_momentum = config.get('BN_momentum', 0.1)
        self.net = UNetFire_P(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=self.kernel_size,
                            recurrent_blocks=recurrent_blocks,
                            BN_momentum=BN_momentum)
        self.num_recurrent_units = self.net.num_recurrent_units
        self.reset_states()

    def reset_states(self):
        self.states = [None] * self.num_recurrent_units

    def forward(self, event_tensor):
        intensity, aolp, dolp, self.states = self.net.forward(event_tensor, self.states)
        return {'intensity': intensity, 'aolp': aolp, 'dolp': dolp}


################# End M 1 ############

################# Start M 2 ############
class UNetFire_P_Split_Input(BaseUNet):
    """
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convgru', base_num_channels=32,
                 num_residual_blocks=2, norm=None, kernel_size=3,
                 recurrent_blocks={'resblock': [0]}, BN_momentum=0.1):
        super(UNetFire_P_Split_Input, self).__init__(num_input_channels=num_input_channels,
                                       num_output_channels=num_output_channels,
                                       skip_type=skip_type,
                                       base_num_channels=base_num_channels,
                                       num_residual_blocks=num_residual_blocks,
                                       norm=norm,
                                       kernel_size=kernel_size)

        self.recurrent_blocks = recurrent_blocks
        self.num_recurrent_units = 0
        self.head = RecurrentConvLayer_Split_Input(self.num_input_channels,
                                       self.base_num_channels,
                                       kernel_size=self.kernel_size,
                                       padding=self.kernel_size // 2,
                                       recurrent_block_type=recurrent_block_type,
                                       norm=self.norm,
                                       BN_momentum=BN_momentum)
        self.num_recurrent_units += 1
        self.resblocks = nn.ModuleList()
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i in range(self.num_residual_blocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                self.resblocks.append(RecurrentResidualLayer(
                    in_channels=self.base_num_channels,
                    out_channels=self.base_num_channels,
                    recurrent_block_type=recurrent_block_type,
                    norm=self.norm,
                    BN_momentum=BN_momentum))
                self.num_recurrent_units += 1
            else:
                self.resblocks.append(ResidualBlock(self.base_num_channels,
                                                    self.base_num_channels,
                                                    norm=self.norm,
                                                    BN_momentum=BN_momentum))
        # intensity
        self.pred_i = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                              self.num_output_channels, kernel_size=1, stride=1, padding=0, activation=None, norm=None)
        self.pred_i.conv2d.bias.data.fill_(0.5)
        # aolp
        self.pred_a = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                                self.num_output_channels, kernel_size=1, stride=1, padding=0, activation=None,
                                norm=None)
        self.pred_a.conv2d.bias.data.fill_(0.5)
        # dolp
        self.pred_d = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                                self.num_output_channels, kernel_size=1, stride=1, padding=0, activation=None,
                                norm=None)
        self.pred_d.conv2d.bias.data.fill_(0.5)

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H/2 x W/2,
                 N x num_output_channels x H/2 x W/2,
                 N x num_output_channels x H/2 x W/2
        """

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        # head
        x, state = self.head(x, prev_states[state_idx])
        state_idx += 1
        states.append(state)

        head_feature_map = x

        # residual blocks
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i, resblock in enumerate(self.resblocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                x, state = resblock(x, prev_states[state_idx])
                state_idx += 1
                states.append(state)
            else:
                x = resblock(x)

        # tail
        y = self.apply_skip_connection(x, head_feature_map)

        intensity = self.pred_i(y)
        aolp = self.pred_a(y)
        dolp = self.pred_d(y)

        return intensity, aolp, dolp, states


class M2(BaseE2VID):
    """
    Use FireNet_legacy to estimate polarization directly.
    """
    def __init__(self, config={}, unet_kwargs={}):
        if unet_kwargs:
            config = unet_kwargs
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        BN_momentum = config.get('BN_momentum', 0.1)
        self.net = UNetFire_P_Split_Input(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=self.kernel_size,
                            recurrent_blocks=recurrent_blocks,
                            BN_momentum=BN_momentum)
        self.num_recurrent_units = self.net.num_recurrent_units
        self.reset_states()

    def reset_states(self):
        self.states = [None] * self.num_recurrent_units

    def forward(self, event_tensor):
        intensity, aolp, dolp, self.states = self.net.forward(event_tensor, self.states)
        return {'intensity': intensity, 'aolp': aolp, 'dolp': dolp}


################# End M 2 ############

################# Start M 4 ############
class UNetFire_P_Split_Input_Warp_AoLP(BaseUNet):
    """
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convgru', base_num_channels=32,
                 num_residual_blocks=2, norm=None, kernel_size=3,
                 recurrent_blocks={'resblock': [0]}, BN_momentum=0.1):
        super(UNetFire_P_Split_Input_Warp_AoLP, self).__init__(num_input_channels=num_input_channels,
                                       num_output_channels=num_output_channels,
                                       skip_type=skip_type,
                                       base_num_channels=base_num_channels,
                                       num_residual_blocks=num_residual_blocks,
                                       norm=norm,
                                       kernel_size=kernel_size)

        self.recurrent_blocks = recurrent_blocks
        self.num_recurrent_units = 0
        self.head = RecurrentConvLayer_Split_Input(self.num_input_channels,
                                       self.base_num_channels,
                                       kernel_size=self.kernel_size,
                                       padding=self.kernel_size // 2,
                                       recurrent_block_type=recurrent_block_type,
                                       norm=self.norm,
                                       BN_momentum=BN_momentum)
        self.num_recurrent_units += 1
        self.resblocks = nn.ModuleList()
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i in range(self.num_residual_blocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                self.resblocks.append(RecurrentResidualLayer(
                    in_channels=self.base_num_channels,
                    out_channels=self.base_num_channels,
                    recurrent_block_type=recurrent_block_type,
                    norm=self.norm,
                    BN_momentum=BN_momentum))
                self.num_recurrent_units += 1
            else:
                self.resblocks.append(ResidualBlock(self.base_num_channels,
                                                    self.base_num_channels,
                                                    norm=self.norm,
                                                    BN_momentum=BN_momentum))
        # intensity
        self.pred_i = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                              self.num_output_channels, kernel_size=1, stride=1, padding=0, activation=None, norm=None)
        self.pred_i.conv2d.bias.data.fill_(0.5)
        # aolp
        self.pred_a = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                                self.num_output_channels, kernel_size=1, stride=1, padding=0, activation=None,
                                norm=None)
        self.pred_a.conv2d.bias.data.fill_(0.5)
        # dolp
        self.pred_d = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                                self.num_output_channels, kernel_size=1, stride=1, padding=0, activation=None,
                                norm=None)
        self.pred_d.conv2d.bias.data.fill_(0.5)

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H/2 x W/2,
                 N x num_output_channels x H/2 x W/2,
                 N x num_output_channels x H/2 x W/2
        """

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        # head
        x, state = self.head(x, prev_states[state_idx])
        state_idx += 1
        states.append(state)

        head_feature_map = x

        # residual blocks
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i, resblock in enumerate(self.resblocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                x, state = resblock(x, prev_states[state_idx])
                state_idx += 1
                states.append(state)
            else:
                x = resblock(x)

        # tail
        y = self.apply_skip_connection(x, head_feature_map)

        intensity = self.pred_i(y)
        aolp = self.pred_a(y)
        aolp = 0.5 * (1 - torch.cos(2 * aolp))
        dolp = self.pred_d(y)

        return intensity, aolp, dolp, states


class M4(BaseE2VID):
    """
    Use FireNet_legacy to estimate polarization directly.
    """
    def __init__(self, config={}, unet_kwargs={}):
        if unet_kwargs:
            config = unet_kwargs
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        BN_momentum = config.get('BN_momentum', 0.1)
        self.net = UNetFire_P_Split_Input_Warp_AoLP(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=self.kernel_size,
                            recurrent_blocks=recurrent_blocks,
                            BN_momentum=BN_momentum)
        self.num_recurrent_units = self.net.num_recurrent_units
        self.reset_states()

    def reset_states(self):
        self.states = [None] * self.num_recurrent_units

    def forward(self, event_tensor):
        intensity, aolp, dolp, self.states = self.net.forward(event_tensor, self.states)
        return {'intensity': intensity, 'aolp': aolp, 'dolp': dolp}


################# End M 2 ############

# def copy_states(states):
#     """
#     LSTM states: [(torch.tensor, torch.tensor), ...]
#     GRU states: [torch.tensor, ...]
#     """
#     if states[0] is None:
#         return copy.deepcopy(states)
#     return recursive_clone(states)
#
#
class ColorNet(BaseModel):
    """
    Split the input events into RGBW channels and feed them to an existing
    recurrent model with states.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.channels = {'R': [slice(0, None, 2), slice(0, None, 2)],
                         'G': [slice(0, None, 2), slice(1, None, 2)],
                         'B': [slice(1, None, 2), slice(1, None, 2)],
                         'W': [slice(1, None, 2), slice(0, None, 2)],
                         'grayscale': [slice(None), slice(None)]}
        self.prev_states = {k: self.model.states for k in self.channels}

    def reset_states(self):
        self.model.reset_states()

    @property
    def num_encoders(self):
        return self.model.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with RGB image taking values in [0, 1], and
                 displacement within event_tensor.
        """
        height, width = event_tensor.shape[-2:]
        crop_halfres = CropParameters(int(width / 2), int(height / 2), self.model.num_encoders)
        crop_fullres = CropParameters(width, height, self.model.num_encoders)
        color_events = {}
        reconstructions_for_each_channel = {}
        for channel, s in self.channels.items():
            color_events = event_tensor[:, :, s[0], s[1]]
            if channel == 'grayscale':
                color_events = crop_fullres.pad(color_events)
            else:
                color_events = crop_halfres.pad(color_events)
            self.model.states = self.prev_states[channel]
            img = self.model(color_events)['image']
            self.prev_states[channel] = self.model.states
            if channel == 'grayscale':
                img = crop_fullres.crop(img)
            else:
                img = crop_halfres.crop(img)
            img = img[0, 0, ...].cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            reconstructions_for_each_channel[channel] = img
        image_bgr = merge_channels_into_color_image(reconstructions_for_each_channel)  # H x W x 3
        return {'image': image_bgr}
#
#
# class WFlowNet(BaseModel):
#     """
#     Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
#     """
#     def __init__(self, unet_kwargs):
#         super().__init__()
#         self.num_bins = unet_kwargs['num_bins']  # legacy
#         self.num_encoders = unet_kwargs['num_encoders']  # legacy
#         self.wnet = WNet(unet_kwargs)
#
#     def reset_states(self):
#         self.wnet.states = [None] * self.wnet.num_encoders
#
#     @property
#     def states(self):
#         return copy_states(self.wnet.states)
#
#     @states.setter
#     def states(self, states):
#         self.wnet.states = states
#
#     def forward(self, event_tensor):
#         """
#         :param event_tensor: N x num_bins x H x W
#         :return: output dict with image taking values in [0,1], and
#                  displacement within event_tensor.
#         """
#         output_dict = self.wnet.forward(event_tensor)
#         return output_dict
#
#
# class FlowNet(BaseModel):
#     """
#     Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
#     """
#     def __init__(self, unet_kwargs):
#         super().__init__()
#         self.num_bins = unet_kwargs['num_bins']  # legacy
#         self.num_encoders = unet_kwargs['num_encoders']  # legacy
#         self.unetflow = UNetFlow(unet_kwargs)
#
#     @property
#     def states(self):
#         return copy_states(self.unetflow.states)
#
#     @states.setter
#     def states(self, states):
#         self.unetflow.states = states
#
#     def reset_states(self):
#         self.unetflow.states = [None] * self.unetflow.num_encoders
#
#     def forward(self, event_tensor):
#         """
#         :param event_tensor: N x num_bins x H x W
#         :return: output dict with image taking values in [0,1], and
#                  displacement within event_tensor.
#         """
#         output_dict = self.unetflow.forward(event_tensor)
#         return output_dict
#
#
# class FlowNetNoRecur(BaseModel):
#     """
#     UNet-like architecture without recurrent units
#     """
#     def __init__(self, unet_kwargs):
#         super().__init__()
#         self.num_bins = unet_kwargs['num_bins']  # legacy
#         self.num_encoders = unet_kwargs['num_encoders']  # legacy
#         self.unetflow = UNetFlowNoRecur(unet_kwargs)
#
#     def reset_states(self):
#         pass
#
#     def forward(self, event_tensor):
#         """
#         :param event_tensor: N x num_bins x H x W
#         :return: output dict with image taking values in [0,1], and
#                  displacement within event_tensor.
#         """
#         output_dict = self.unetflow.forward(event_tensor)
#         return output_dict
#
#
# class E2VIDRecurrent(BaseModel):
#     """
#     Compatible with E2VID_lightweight
#     Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
#     """
#     def __init__(self, unet_kwargs):
#         super().__init__()
#         self.num_bins = unet_kwargs['num_bins']  # legacy
#         self.num_encoders = unet_kwargs['num_encoders']  # legacy
#         self.unetrecurrent = UNetRecurrent(unet_kwargs)
#
#     @property
#     def states(self):
#         return copy_states(self.unetrecurrent.states)
#
#     @states.setter
#     def states(self, states):
#         self.unetrecurrent.states = states
#
#     def reset_states(self):
#         self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
#
#     def forward(self, event_tensor):
#         """
#         :param event_tensor: N x num_bins x H x W
#         :return: output dict with image taking values in [0,1], and
#                  displacement within event_tensor.
#         """
#         output_dict = self.unetrecurrent.forward(event_tensor)
#         return output_dict
#
#
# class EVFlowNet(BaseModel):
#     """
#     Model from the paper: "EV-FlowNet: Self-Supervised Optical Flow for Event-based Cameras", Zhu et al. 2018.
#     Pytorch adaptation of https://github.com/daniilidis-group/EV-FlowNet/blob/master/src/model.py (may differ slightly)
#     """
#     def __init__(self, unet_kwargs):
#         super().__init__()
#         # put 'hardcoded' EVFlowNet parameters here
#         EVFlowNet_kwargs = {
#             'base_num_channels': 32, # written as '64' in EVFlowNet tf code
#             'num_encoders': 4,
#             'num_residual_blocks': 2,  # transition
#             'num_output_channels': 2,  # (x, y) displacement
#             'skip_type': 'concat',
#             'norm': None,
#             'use_upsample_conv': True,
#             'kernel_size': 3,
#             'channel_multiplier': 2
#             }
#         unet_kwargs.update(EVFlowNet_kwargs)
#
#         self.num_bins = unet_kwargs['num_bins']  # legacy
#         self.num_encoders = unet_kwargs['num_encoders']  # legacy
#         self.unet = UNet(unet_kwargs)
#
#     def reset_states(self):
#         pass
#
#     def forward(self, event_tensor):
#         """
#         :param event_tensor: N x num_bins x H x W
#         :return: output dict with N x 2 X H X W (x, y) displacement within event_tensor.
#         """
#         flow = self.unet.forward(event_tensor)
#         # to make compatible with our training/inference code that expects an image, make a dummy image.
#         return {'flow': flow, 'image': 0 * flow[..., 0:1, :, :]}
#
#
# class FireNet(BaseModel):
#     """
#     Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
#     The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
#     However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
#     """
#     def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
#         super().__init__()
#         if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
#             num_bins = unet_kwargs.get('num_bins', num_bins)
#             base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
#             kernel_size = unet_kwargs.get('kernel_size', kernel_size)
#         self.num_bins = num_bins
#         padding = kernel_size // 2
#         self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
#         self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R1 = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2 = ResidualBlock(base_num_channels, base_num_channels)
#         self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
#         self.num_encoders = 0  # needed by image_reconstructor.py
#         self.num_recurrent_units = 2
#         self.reset_states()
#
#     @property
#     def states(self):
#         return copy_states(self._states)
#
#     @states.setter
#     def states(self, states):
#         self._states = states
#
#     def reset_states(self):
#         self._states = [None] * self.num_recurrent_units
#
#     def forward(self, x):
#         """
#         :param x: N x num_input_channels x H x W event tensor
#         :return: N x num_output_channels x H x W image
#         """
#         x = self.head(x)
#         x = self.G1(x, self._states[0])
#         self._states[0] = x
#         x = self.R1(x)
#         x = self.G2(x, self._states[1])
#         self._states[1] = x
#         x = self.R2(x)
#         return {'image': self.pred(x)}
#         # return {'image': nn.Sigmoid()(self.pred(x))}
#
#
# class FireNet_P(BaseModel):
#     """
#     Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
#     The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
#     However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
#
#     Revised by Haiyang Mei to predict polarization information directly.
#
#     """
#     def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
#         super().__init__()
#         if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
#             num_bins = unet_kwargs.get('num_bins', num_bins)
#             base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
#             kernel_size = unet_kwargs.get('kernel_size', kernel_size)
#             final_activation = unet_kwargs.get('final_activation', None)
#             if final_activation == "":
#                 final_activation = None
#         self.num_bins = num_bins
#         padding = kernel_size // 2
#         self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
#         self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R1 = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2 = ResidualBlock(base_num_channels, base_num_channels)
#         self.pred_intensity = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=2, padding=1, activation=final_activation)
#         self.pred_aolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=2, padding=1, activation=final_activation)
#         self.pred_dolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=2, padding=1, activation=final_activation)
#         self.num_encoders = 0  # needed by image_reconstructor.py
#         self.num_recurrent_units = 2
#         self.reset_states()
#
#     @property
#     def states(self):
#         return copy_states(self._states)
#
#     @states.setter
#     def states(self, states):
#         self._states = states
#
#     def reset_states(self):
#         self._states = [None] * self.num_recurrent_units
#
#     def forward(self, x):
#         """
#         :param x: N x num_input_channels x H x W event tensor
#         :return: N x num_output_channels x H x W image
#         """
#         x = self.head(x)
#         x = self.G1(x, self._states[0])
#         self._states[0] = x
#         x = self.R1(x)
#         x = self.G2(x, self._states[1])
#         self._states[1] = x
#         x = self.R2(x)
#         return {'intensity': self.pred_intensity(x), 'aolp': self.pred_aolp(x), 'dolp': self.pred_dolp(x)}
#
#
# class V1(BaseModel):
#     """
#     v1: estimate polarization directly.
#
#     """
#     def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
#         super(V1, self).__init__()
#         if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
#             num_bins = unet_kwargs.get('num_bins', num_bins)
#             base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
#             kernel_size = unet_kwargs.get('kernel_size', kernel_size)
#             final_activation = unet_kwargs.get('final_activation', None)
#             if final_activation == "":
#                 final_activation = None
#         self.num_bins = num_bins
#         padding = kernel_size // 2
#         self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
#         self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R1 = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2 = ResidualBlock(base_num_channels, base_num_channels)
#         self.pred_intensity = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=2, padding=1, activation=final_activation)
#         self.pred_aolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=2, padding=1, activation=final_activation)
#         self.pred_dolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=2, padding=1, activation=final_activation)
#         self.num_encoders = 0  # needed by image_reconstructor.py
#         self.num_recurrent_units = 2
#         self.reset_states()
#
#     @property
#     def states(self):
#         return copy_states(self._states)
#
#     @states.setter
#     def states(self, states):
#         self._states = states
#
#     def reset_states(self):
#         self._states = [None] * self.num_recurrent_units
#
#     def forward(self, x):
#         """
#         :param x: N x num_input_channels x H x W event tensor
#         :return: N x num_output_channels x H x W image
#         """
#         x = self.head(x)
#         x = self.G1(x, self._states[0])
#         self._states[0] = x
#         x = self.R1(x)
#         x = self.G2(x, self._states[1])
#         self._states[1] = x
#         x = self.R2(x)
#         return {'intensity': self.pred_intensity(x), 'aolp': self.pred_aolp(x), 'dolp': self.pred_dolp(x)}
#
#
# class V2(BaseModel):
#     """
#     v2: estimate polarization directly, split input and concat four types of features.
#
#     """
#     def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
#         super(V2, self).__init__()
#         if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
#             num_bins = unet_kwargs.get('num_bins', num_bins)
#             base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
#             kernel_size = unet_kwargs.get('kernel_size', kernel_size)
#             final_activation = unet_kwargs.get('final_activation', None)
#             if final_activation == "":
#                 final_activation = None
#         self.num_bins = num_bins
#         padding = kernel_size // 2
#         self.head_90 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.head_45 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.head_135 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.head_0 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R1 = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2 = ResidualBlock(base_num_channels, base_num_channels)
#         self.pred_intensity = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=final_activation)
#         # self.pred_aolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=final_activation)
#         # for V5
#         self.pred_aolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
#         self.pred_dolp = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=final_activation)
#         self.num_encoders = 0  # needed by image_reconstructor.py
#         self.num_recurrent_units = 2
#         self.reset_states()
#
#     @property
#     def states(self):
#         return copy_states(self._states)
#
#     @states.setter
#     def states(self, states):
#         self._states = states
#
#     def reset_states(self):
#         self._states = [None] * self.num_recurrent_units
#
#     def forward(self, x):
#         """
#         :param x: N x num_input_channels x H x W event tensor
#         :return: N x num_output_channels x H x W image
#         """
#         x_90 = x[:, :, 0::2, 0::2]
#         x_45 = x[:, :, 0::2, 1::2]
#         x_135 = x[:, :, 1::2, 0::2]
#         x_0 = x[:, :, 1::2, 1::2]
#
#         x_90 = self.head_90(x_90)
#         x_45 = self.head_45(x_45)
#         x_135 = self.head_135(x_135)
#         x_0 = self.head_0(x_0)
#
#         x = torch.cat([x_90, x_45, x_135, x_0], 1)
#         x = self.G1(x, self._states[0])
#         self._states[0] = x
#         x = self.R1(x)
#         x = self.G2(x, self._states[1])
#         self._states[1] = x
#         x = self.R2(x)
#         # return {'intensity': self.pred_intensity(x), 'aolp': self.pred_aolp(x), 'dolp': self.pred_dolp(x)}
#         # for v5
#         return {'intensity': self.pred_intensity(x), 'aolp': 0.5*(1-torch.cos(2*self.pred_aolp(x))), 'dolp': self.pred_dolp(x)}
#
#
# class V3(BaseModel):
#     """
#     v3: estimate polarization directly,
#     split input and concat four types of features,
#     use three branches to estimate polarization.
#
#     """
#     def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
#         super(V3, self).__init__()
#         if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
#             num_bins = unet_kwargs.get('num_bins', num_bins)
#             base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
#             kernel_size = unet_kwargs.get('kernel_size', kernel_size)
#             final_activation = unet_kwargs.get('final_activation', None)
#             if final_activation == "":
#                 final_activation = None
#         self.num_bins = num_bins
#         padding = kernel_size // 2
#         self.head_90 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.head_45 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.head_135 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.head_0 = ConvLayer(self.num_bins, int(base_num_channels / 4), kernel_size, padding=padding)
#         self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R1 = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2_I = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2_I = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2_A = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2_A = ResidualBlock(base_num_channels, base_num_channels)
#         self.G2_D = ConvGRU(base_num_channels, base_num_channels, kernel_size)
#         self.R2_D = ResidualBlock(base_num_channels, base_num_channels)
#         self.pred_I = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=final_activation)
#         # self.pred_A = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=final_activation)
#         self.pred_A = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=None)
#         self.pred_D = ConvLayer(base_num_channels, out_channels=1, kernel_size=3, stride=1, padding=1, activation=final_activation)
#         self.num_encoders = 0  # needed by image_reconstructor.py
#         self.num_recurrent_units = 4
#         self.reset_states()
#
#     @property
#     def states(self):
#         return copy_states(self._states)
#
#     @states.setter
#     def states(self, states):
#         self._states = states
#
#     def reset_states(self):
#         self._states = [None] * self.num_recurrent_units
#
#     def forward(self, x):
#         """
#         :param x: N x num_input_channels x H x W event tensor
#         :return: N x num_output_channels x H x W image
#         """
#         x_90 = x[:, :, 0::2, 0::2]
#         x_45 = x[:, :, 0::2, 1::2]
#         x_135 = x[:, :, 1::2, 0::2]
#         x_0 = x[:, :, 1::2, 1::2]
#
#         x_90 = self.head_90(x_90)
#         x_45 = self.head_45(x_45)
#         x_135 = self.head_135(x_135)
#         x_0 = self.head_0(x_0)
#
#         x = torch.cat([x_90, x_45, x_135, x_0], 1)
#         x = self.G1(x, self._states[0])
#         self._states[0] = x
#         x = self.R1(x)
#
#         # intensity branch
#         x_I = self.G2_I(x, self._states[1])
#         self._states[1] = x_I
#         x_I = self.R2_I(x_I)
#
#         # aolp branch
#         x_A = self.G2_A(x, self._states[2])
#         self._states[2] = x_A
#         x_A = self.R2_A(x_A)
#
#         # dolp branch
#         x_D = self.G2_D(x, self._states[3])
#         self._states[3] = x_D
#         x_D = self.R2_D(x_D)
#
#         return {'intensity': self.pred_I(x_I), 'aolp': 0.5*(1-torch.cos(2*self.pred_A(x_A))), 'dolp': self.pred_D(x_D)}
