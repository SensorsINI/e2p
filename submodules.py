"""
 @Time    : 21.09.22 14:17
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : submodules_v.py
 @Function:
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = self.relu(conv1)

        conv2 = self.conv2(relu1)
        relu2 = self.relu(conv2 + x)

        return relu2


class ResidualBlock_BN(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock_BN, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2 + x)
        relu2 = self.relu(bn2)

        return relu2


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class Average(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Average, self).__init__()
        self.conv1_x = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        self.conv3_x = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        self.conv1_y = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        self.conv3_y = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        self.relu = nn.ReLU()

        self.fusion = nn.Conv2d(dim_out * 4, dim_out, 1, 1, 0)

    def forward(self, x, y):
        conv1_x = self.conv1_x(x)
        conv3_x = self.conv3_x(x)

        conv1_y = self.conv1_y(y)
        conv3_y = self.conv3_y(y)

        sum1 = self.relu(conv1_x + conv1_y)
        sum2 = self.relu(conv1_x + conv3_y)
        sum3 = self.relu(conv3_x + conv1_y)
        sum4 = self.relu(conv3_x + conv3_y)

        fusion = self.fusion(torch.cat((sum1, sum2, sum3, sum4), 1))

        return fusion


class Contrast(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Contrast, self).__init__()
        self.conv1_x = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        self.conv3_x = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        self.conv1_y = nn.Conv2d(dim_in, dim_out, 1, 1, 0)
        self.conv3_y = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        self.relu = nn.ReLU()

        self.fusion = nn.Conv2d(dim_out * 4, dim_out, 1, 1, 0)

    def forward(self, x, y):
        conv1_x = self.conv1_x(x)
        conv3_x = self.conv3_x(x)

        conv1_y = self.conv1_y(y)
        conv3_y = self.conv3_y(y)

        contrast1 = self.relu(conv1_x - conv1_y)
        contrast2 = self.relu(conv1_x - conv3_y)
        contrast3 = self.relu(conv3_x - conv1_y)
        contrast4 = self.relu(conv3_x - conv3_y)

        fusion = self.relu(self.fusion(torch.cat((contrast1, contrast2, contrast3, contrast4), 1)))

        return fusion


class Coupled_Layer(nn.Module):
    def __init__(self, n_feats=16, coupled_number=12, kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size

        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i90_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i45_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i135_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i0_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i90_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i45_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i135_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i0_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i90_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i45_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i135_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i0_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i90_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i45_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i135_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i0_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_i90, feat_i45, feat_i135, feat_i0):
        # i90
        shortCut_i90 = feat_i90
        feat_i90 = F.conv2d(feat_i90,
                            torch.cat([self.kernel_shared_1, self.kernel_i90_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i90_1], dim=0), padding=1)
        feat_i90 = F.relu(feat_i90, inplace=True)
        feat_i90 = F.conv2d(feat_i90,
                            torch.cat([self.kernel_shared_2, self.kernel_i90_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i90_2], dim=0), padding=1)
        feat_i90 = F.relu(feat_i90 + shortCut_i90, inplace=True)

        # i45
        shortCut_i45 = feat_i45
        feat_i45 = F.conv2d(feat_i45,
                            torch.cat([self.kernel_shared_1, self.kernel_i45_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i45_1], dim=0), padding=1)
        feat_i45 = F.relu(feat_i45, inplace=True)
        feat_i45 = F.conv2d(feat_i45,
                            torch.cat([self.kernel_shared_2, self.kernel_i45_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i45_2], dim=0), padding=1)
        feat_i45 = F.relu(feat_i45 + shortCut_i45, inplace=True)

        # i135
        shortCut_i135 = feat_i135
        feat_i135 = F.conv2d(feat_i135,
                            torch.cat([self.kernel_shared_1, self.kernel_i135_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i135_1], dim=0), padding=1)
        feat_i135 = F.relu(feat_i135, inplace=True)
        feat_i135 = F.conv2d(feat_i135,
                            torch.cat([self.kernel_shared_2, self.kernel_i135_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i135_2], dim=0), padding=1)
        feat_i135 = F.relu(feat_i135 + shortCut_i135, inplace=True)

        # i0
        shortCut_i0 = feat_i0
        feat_i0 = F.conv2d(feat_i0,
                            torch.cat([self.kernel_shared_1, self.kernel_i0_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i0_1], dim=0), padding=1)
        feat_i0 = F.relu(feat_i0, inplace=True)
        feat_i0 = F.conv2d(feat_i0,
                            torch.cat([self.kernel_shared_2, self.kernel_i0_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i0_2], dim=0), padding=1)
        feat_i0 = F.relu(feat_i0 + shortCut_i0, inplace=True)

        return feat_i90, feat_i45, feat_i135, feat_i0


class JR(nn.Module):
    """
    Joint Residual.
    """
    def __init__(self, dim, n_layer=2):
        super(JR, self).__init__()
        dim_joint = int(0.75 * dim)
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer(dim, dim_joint) for i in range(self.n_layer)])

    def forward(self, feat_i90, feat_i45, feat_i135, feat_i0):
        for layer in self.coupled_feat_extractor:
            feat_i90, feat_i45, feat_i135, feat_i0 = layer(feat_i90, feat_i45, feat_i135, feat_i0)

        return feat_i90, feat_i45, feat_i135, feat_i0


class Coupled_Layer_IAD(nn.Module):
    def __init__(self, n_feats=32, coupled_number=8, kernel_size=3):
        super(Coupled_Layer_IAD, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size

        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_a_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_d_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_a_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_d_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_a_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_d_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_a_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_d_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_i, feat_a, feat_d):
        # i
        shortCut_i = feat_i
        feat_i = F.conv2d(feat_i,
                            torch.cat([self.kernel_shared_1, self.kernel_i_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i_1], dim=0), padding=1)
        feat_i = F.relu(feat_i, inplace=True)
        feat_i = F.conv2d(feat_i,
                            torch.cat([self.kernel_shared_2, self.kernel_i_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i_2], dim=0), padding=1)
        feat_i = F.relu(feat_i + shortCut_i, inplace=True)

        # a
        shortCut_a = feat_a
        feat_a = F.conv2d(feat_a,
                            torch.cat([self.kernel_shared_1, self.kernel_a_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_a_1], dim=0), padding=1)
        feat_a = F.relu(feat_a, inplace=True)
        feat_a = F.conv2d(feat_a,
                            torch.cat([self.kernel_shared_2, self.kernel_a_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_a_2], dim=0), padding=1)
        feat_a = F.relu(feat_a + shortCut_a, inplace=True)

        # d
        shortCut_d = feat_d
        feat_d = F.conv2d(feat_d,
                            torch.cat([self.kernel_shared_1, self.kernel_d_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_d_1], dim=0), padding=1)
        feat_d = F.relu(feat_d, inplace=True)
        feat_d = F.conv2d(feat_d,
                            torch.cat([self.kernel_shared_2, self.kernel_d_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_d_2], dim=0), padding=1)
        feat_d = F.relu(feat_d + shortCut_d, inplace=True)

        return feat_i, feat_a, feat_d


class JR_IAD(nn.Module):
    """
    Joint Residual for IAD.
    """
    def __init__(self, dim, n_layer=2):
        super(JR_IAD, self).__init__()
        dim_joint = int(0.25 * dim)
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer_IAD(dim, dim_joint) for i in range(self.n_layer)])

    def forward(self, feat_i, feat_a, feat_d):
        for layer in self.coupled_feat_extractor:
            feat_i, feat_a, feat_d = layer(feat_i, feat_a, feat_d)

        return feat_i, feat_a, feat_d


class Coupled_Layer_BN(nn.Module):
    def __init__(self, n_feats=16, coupled_number=12, kernel_size=3):
        super(Coupled_Layer_BN, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size

        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i90_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i45_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i135_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i0_1 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i90_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i45_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i135_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_i0_2 = nn.Parameter(nn.init.kaiming_uniform_(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i90_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i45_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i135_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i0_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_i90_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i45_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i135_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_i0_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bn_i90_1 = nn.BatchNorm2d(n_feats)
        self.bn_i90_2 = nn.BatchNorm2d(n_feats)
        self.bn_i45_1 = nn.BatchNorm2d(n_feats)
        self.bn_i45_2 = nn.BatchNorm2d(n_feats)
        self.bn_i135_1 = nn.BatchNorm2d(n_feats)
        self.bn_i135_2 = nn.BatchNorm2d(n_feats)
        self.bn_i0_1 = nn.BatchNorm2d(n_feats)
        self.bn_i0_2 = nn.BatchNorm2d(n_feats)

    def forward(self, feat_i90, feat_i45, feat_i135, feat_i0):
        # i90
        shortCut_i90 = feat_i90
        feat_i90 = F.conv2d(feat_i90,
                            torch.cat([self.kernel_shared_1, self.kernel_i90_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i90_1], dim=0), padding=1)
        feat_i90 = self.bn_i90_1(feat_i90)
        feat_i90 = F.relu(feat_i90, inplace=True)
        feat_i90 = F.conv2d(feat_i90,
                            torch.cat([self.kernel_shared_2, self.kernel_i90_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i90_2], dim=0), padding=1)
        feat_i90 = self.bn_i90_2(feat_i90 + shortCut_i90)
        feat_i90 = F.relu(feat_i90, inplace=True)

        # i45
        shortCut_i45 = feat_i45
        feat_i45 = F.conv2d(feat_i45,
                            torch.cat([self.kernel_shared_1, self.kernel_i45_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i45_1], dim=0), padding=1)
        feat_i45 = self.bn_i45_1(feat_i45)
        feat_i45 = F.relu(feat_i45, inplace=True)
        feat_i45 = F.conv2d(feat_i45,
                            torch.cat([self.kernel_shared_2, self.kernel_i45_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i45_2], dim=0), padding=1)
        feat_i45 = self.bn_i45_2(feat_i45 + shortCut_i45)
        feat_i45 = F.relu(feat_i45, inplace=True)

        # i135
        shortCut_i135 = feat_i135
        feat_i135 = F.conv2d(feat_i135,
                            torch.cat([self.kernel_shared_1, self.kernel_i135_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i135_1], dim=0), padding=1)
        feat_i135 = self.bn_i135_1(feat_i135)
        feat_i135 = F.relu(feat_i135, inplace=True)
        feat_i135 = F.conv2d(feat_i135,
                            torch.cat([self.kernel_shared_2, self.kernel_i135_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i135_2], dim=0), padding=1)
        feat_i135 = self.bn_i135_2(feat_i135 + shortCut_i135)
        feat_i135 = F.relu(feat_i135, inplace=True)

        # i0
        shortCut_i0 = feat_i0
        feat_i0 = F.conv2d(feat_i0,
                            torch.cat([self.kernel_shared_1, self.kernel_i0_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_i0_1], dim=0), padding=1)
        feat_i0 = self.bn_i0_1(feat_i0)
        feat_i0 = F.relu(feat_i0, inplace=True)
        feat_i0 = F.conv2d(feat_i0,
                            torch.cat([self.kernel_shared_2, self.kernel_i0_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_i0_2], dim=0), padding=1)
        feat_i0 = self.bn_i0_2(feat_i0 + shortCut_i0)
        feat_i0 = F.relu(feat_i0, inplace=True)

        return feat_i90, feat_i45, feat_i135, feat_i0


class JR_BN(nn.Module):
    """
    Joint Residual with batch normalization.
    """
    def __init__(self, dim, n_layer=2):
        super(JR_BN, self).__init__()
        dim_joint = int(0.75 * dim)
        self.n_layer = n_layer
        self.coupled_feat_extractor = nn.ModuleList([Coupled_Layer_BN(dim, dim_joint) for i in range(self.n_layer)])

    def forward(self, feat_i90, feat_i45, feat_i135, feat_i0):
        for layer in self.coupled_feat_extractor:
            feat_i90, feat_i45, feat_i135, feat_i0 = layer(feat_i90, feat_i45, feat_i135, feat_i0)

        return feat_i90, feat_i45, feat_i135, feat_i0


class RPPP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.ReLU(True))

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.ReLU(True))

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.ReLU(True))

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class RPPP_BN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.pad1 = nn.ZeroPad2d((1, 0, 1, 0))
        self.pad2 = nn.ZeroPad2d((0, 1, 1, 0))
        self.pad3 = nn.ZeroPad2d((1, 0, 0, 1))
        self.pad4 = nn.ZeroPad2d((0, 1, 0, 1))

        dim_mid = dim_out // 4
        self.pattern1 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.BatchNorm2d(dim_mid), nn.ReLU(True))
        self.pattern2 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.BatchNorm2d(dim_mid), nn.ReLU(True))
        self.pattern3 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.BatchNorm2d(dim_mid), nn.ReLU(True))
        self.pattern4 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 3, stride=2, padding=0), nn.BatchNorm2d(dim_mid), nn.ReLU(True))

        self.pattern0 = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 2, stride=2, padding=0), nn.BatchNorm2d(dim_mid), nn.ReLU(True))

        self.conv = nn.Sequential(nn.Conv2d(dim_mid * 5, dim_out, 1, 1, 0), nn.BatchNorm2d(dim_out), nn.ReLU(True))

    def forward(self, x):
        x_0 = self.pattern0(x)

        x_1 = self.pattern1(self.pad1(x))
        x_2 = self.pattern2(self.pad2(x))
        x_3 = self.pattern3(self.pad3(x))
        x_4 = self.pattern4(self.pad4(x))

        x_c = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)

        x_conv = self.conv(x_c)

        return x_conv


class Enhance(nn.Module):
    """
    ESA
    """

    def __init__(self, dim):
        super(Enhance, self).__init__()
        dim1 = int(dim / 4)
        dim2 = int(dim / 2)
        dim3 = int(dim / 1)

        self.e1 = nn.Sequential(nn.Conv2d(1, dim1, 4, 2, 1), nn.ReLU(True))
        self.e2 = nn.Sequential(nn.Conv2d(dim1, dim2, 4, 2, 1), nn.ReLU(True))
        self.e3 = nn.Sequential(nn.Conv2d(dim2, dim3, 4, 2, 1), nn.ReLU(True))

        self.d3 = nn.Sequential(nn.Conv2d(dim3, dim2, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        self.d2 = nn.Sequential(nn.Conv2d(dim2, dim1, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        self.d1 = nn.Sequential(nn.Conv2d(dim1, 1, 3, 1, 1), nn.UpsamplingBilinear2d(scale_factor=2))

        self.relu = nn.ReLU(True)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        d3 = self.d3(e3)
        d2 = self.d2(self.relu(d3 + e2))
        d1 = self.d1(self.relu(d2 + e1))

        output = x + d1

        return output


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class Fusion_Weight(nn.Module):
    """
    ESA to generate fusion weight.
    """
    def __init__(self, dim):
        super(Fusion_Weight, self).__init__()
        dim_double = dim * 2
        self.conv1 = nn.Sequential(nn.Conv2d(dim_double, dim, 1, 1, 0), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 4, 2, 1), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True))
        self.conv4 = nn.Conv2d(dim, dim_double, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        max_pool = F.max_pool2d(conv2, kernel_size=4, stride=4, padding=0)
        conv3 = self.conv3(max_pool)
        up = F.interpolate(conv3, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        conv4 = self.conv4(conv1 + up)
        weight = self.sigmoid(conv4)

        return weight


class Fusion_Weight_IAD(nn.Module):
    """
    ESA to generate fusion weight.
    """
    def __init__(self, dim):
        super(Fusion_Weight_IAD, self).__init__()
        dim_triple = dim * 3
        self.conv1 = nn.Sequential(nn.Conv2d(dim_triple, dim, 1, 1, 0), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 4, 2, 1), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True))
        self.conv4 = nn.Conv2d(dim, dim_triple, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        max_pool = F.max_pool2d(conv2, kernel_size=4, stride=4, padding=0)
        conv3 = self.conv3(max_pool)
        up = F.interpolate(conv3, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        conv4 = self.conv4(conv1 + up)
        weight = self.sigmoid(conv4)

        return weight
