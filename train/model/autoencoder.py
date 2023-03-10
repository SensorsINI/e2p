"""
 @Time    : 09.09.22 19:16
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : ae
 @File    : autoencoder.py
 @Function:
 
"""
import torch.nn as nn


class RB(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, down_ch=False):
        super(RB, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))

        if down_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu_out = relu(*param)

    def forward(self, x):

        identity = x

        x = self.conv(x)
        x = x + self.shortcut(identity)
        x = self.relu_out(x)

        return x


class Up(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, bilinear=True):
        super(Up, self).__init__()

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_ch),
                relu(*param)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                norm_layer(out_ch),
                relu(*param)
            )

    def forward(self, x):
        x = self.up(x)
        return x


###################################################################
# ########################## NETWORK ##############################
###################################################################
class AutoEncoder(nn.Module):
    def __init__(self, leaky=True, base_channel=32, norm_layer=nn.BatchNorm2d):
        super(AutoEncoder, self).__init__()
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]

        self.filters = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8]

        self.Conv_input = nn.Sequential(nn.Conv2d(1, self.filters[0], kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(self.filters[0]),
                                        relu(*param))

        self.Conv1 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=False)
        self.Conv2 = RB(self.filters[0], self.filters[1], norm_layer, leaky=leaky, down_ch=True)
        self.Conv3 = RB(self.filters[1], self.filters[2], norm_layer, leaky=leaky, down_ch=True)
        self.Conv4 = RB(self.filters[2], self.filters[3], norm_layer, leaky=leaky, down_ch=True)

        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, bilinear=True)
        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, down_ch=True)

        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, bilinear=True)
        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, down_ch=True)

        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, bilinear=True)
        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, down_ch=True)

        self.Conv = nn.Conv2d(self.filters[0], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input

        x_in = x

        x_in = self.Conv_input(x_in)

        e1 = self.Conv1(x_in)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        x = out

        return e1, e2, e3, e4, x
