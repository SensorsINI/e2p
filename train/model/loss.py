import torch
import torch.nn.functional as F
import numpy as np
# local modules
from train.PerceptualSimilarity import models
from train.utils import loss
from train.model.mhy_ssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn as nn
import train.dct as dct


class combined_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred_img, pred_flow, target_img, target_flow):
        """
        image is tensor of N x 2 x H x W, flow of N x 2 x H x W
        These are concatenated, as perceptualLoss expects N x 3 x H x W.
        """
        pred = torch.cat([pred_img, pred_flow], dim=1)
        target = torch.cat([target_img, target_flow], dim=1)
        dist = self.loss(pred, target, normalize=False)
        return dist * self.weight


class warping_flow_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.warping_flow_loss
        self.weight = weight
        self.L0 = L0
        self.default_return = None

    def __call__(self, i, image1, flow):
        """
        flow is from image0 to image1 (reversed when passed to
        warping_flow_loss function)
        """
        loss = self.default_return if i < self.L0 else self.weight * self.loss(
                self.image0, image1, -flow)
        self.image0 = image1
        return loss


class voxel_warp_flow_loss():
    def __init__(self, weight=1.0):
        self.loss = loss.voxel_warping_flow_loss
        self.weight = weight

    def __call__(self, voxel, displacement, output_images=False):
        """
        Warp the voxel grid by the displacement map. Variance 
        of resulting image is loss
        """
        loss = self.loss(voxel, displacement, output_images)
        if output_images:
            loss = (self.weight * loss[0], loss[1])
        else:
            loss *= self.weight
        return loss


class flow_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target):
        """
        pred and target are Tensors with shape N x 2 x H x W
        PerceptualLoss expects N x 3 x H x W.
        """
        dist_x = self.loss(pred[:, 0:1, :, :], target[:, 0:1, :, :], normalize=False)
        dist_y = self.loss(pred[:, 1:2, :, :], target[:, 1:2, :, :], normalize=False)
        return (dist_x + dist_y) / 2 * self.weight


class flow_l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


# keep for compatibility
flow_loss = flow_l1_loss


class perceptual_loss():
    # alex or vgg
    def __init__(self, weight=1.0, net='alex', use_gpu=True):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return self.weight * dist.mean()


class mse_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class mse_loss_aolp():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class mse_circular_loss():
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def loss(self, pred, target):
        return torch.mean(2-2*(torch.cos(pred - target)))

    def __call__(self, pred, target):
        mse_circular = self.loss(pred, target)
        return self.weight * mse_circular

class abs_sin_loss():
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def loss(self, pred, target):
        pred = pred*np.pi
        target = target*np.pi
        return torch.mean(torch.abs(torch.sin(pred - target)))  #changing it to -pi/2 to pi/2 first

    def __call__(self, pred, target):
        abs_sin_loss = self.loss(pred, target)
        return self.weight * abs_sin_loss

class masked_abs_sin_loss():
    def __init__(self, weight=1.0, threshold=None):
        self.weight = weight
        self.threshold = threshold
    
    def loss(self, pred, target, dolp_gt):
        pred = pred*np.pi
        target = target*np.pi
        mask = dolp_gt > self.threshold
        return torch.mean(torch.abs(torch.sin(pred*mask - target*mask)))  #changing it to -pi/2 to pi/2 first

    def __call__(self, pred, target, dolp_gt):
        abs_sin_loss = self.loss(pred, target, dolp_gt)
        return self.weight * abs_sin_loss
    

class masked_sqrt_cos_loss():
    def __init__(self, weight=1.0, threshold=None):
        self.weight = weight
        self.threshold = threshold
    
    def loss(self, pred, target, dolp_gt):
        pred = pred*np.pi
        target = target*np.pi
        mask = dolp_gt > self.threshold
        return torch.mean(torch.sqrt(2-2*torch.cos(pred*mask - target*mask) + 1e-8)) 

    def __call__(self, pred, target, dolp_gt):
        sqrt_cos_loss = self.loss(pred, target, dolp_gt)
        return self.weight * sqrt_cos_loss 

class masked_aolp_sin_cos_mse_loss():
    def __init__(self, weight=1.0, threshold=None):
        self.weight = weight
        self.threshold = threshold
    
    def loss(self, pred, target, dolp_gt):
        # get dolp mask with ground truth dolp, the aolp loss is only calculated on the pixels with large enough dolp
        mask = dolp_gt > self.threshold
        # pred is (batch, 2), target is (batch, 1) and is in [0,1]
        # first shift the target in [0,1] to (-pi, pi)
        target = target * 2 * np.pi - np.pi
        # then get sin and cos of target
        target_sin = torch.sin(target)
        target_cos = torch.cos(target)
        pred_sin = pred[:,0:1]
        pred_cos = pred[:,1:2]
        # print(f"shape target:{target.shape} target_sin:{target_sin.shape} pred:{pred.shape} pred_sin:{pred_sin.shape}")
        return F.mse_loss(target_sin*mask, pred_sin*mask) \
            + F.mse_loss(target_cos*mask, pred_cos*mask)

    def __call__(self, pred, target, dolp_gt):
        sqrt_cos_loss = self.loss(pred, target, dolp_gt)
        return self.weight * sqrt_cos_loss     

class l2_dw_loss():
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, pred, target_a, target_d, flag):
        if flag == 'aolp':
            loss = (1 + target_d) * (pred - target_a) ** 2
        else:
            loss = 2 * (pred - target_d) ** 2

        return self.weight * torch.mean(loss)


class temporal_consistency_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.temporal_consistency_loss
        self.weight = weight
        self.L0 = L0

    def __call__(self, i, image1, processed1, flow, output_images=False):
        """
        flow is from image0 to image1 (reversed when passed to
        temporal_consistency_loss function)
        """
        if i >= self.L0:
            loss = self.loss(self.image0, image1, self.processed0, processed1,
                             -flow, output_images=output_images)
            if output_images:
                loss = (self.weight * loss[0], loss[1])
            else:
                loss *= self.weight
        else:
            loss = None
        self.image0 = image1
        self.processed0 = processed1
        return loss


# added by Haiyang Mei
class l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class ssim_loss():
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, pred, target):
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)

        return self.weight * ssim_loss

class ms_ssim_loss():
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, pred, target):
        ms_ssim_module = MS_SSIM(data_range=1.0, win_size=7, size_average=True, channel=1, weights=[1, 1, 1, 1])
        ms_ssim_loss = 1 - ms_ssim_module(pred, target)

        return self.weight * ms_ssim_loss

# class cos_loss():
#     def __init__(self, weight=1.0):
#         self.weight = weight
#
#     def __call__(self, pred, target):
#         pred_scale = 2 * pred * torch.pi
#         target_scale = target * torch.pi
#         diff = pred_scale - target_scale
#         loss = 1 - torch.cos(2 * diff)
#         weighted_loss = self.weight * torch.mean(loss)
#
#         return weighted_loss

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, weight=1.0):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
        self.weight = weight

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps)
        loss = torch.mean(error)
        return self.weight * loss

class frequency_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        # self.loss = F.mse_loss
        # self.loss = L1_Charbonnier_loss()
        self.weight = weight

    def __call__(self, pred, target):
        pred_dct = torch.fft.rfft2(pred, norm='ortho')
        # print(pred.shape)
        # print(pred_dct.shape)
        target_dct = torch.fft.rfft2(target, norm='ortho')

        # return self.weight * (self.loss(pred_dct.real, target_dct.real) + self.loss(pred_dct.imag, target_dct.imag))
        return self.weight * (self.loss(pred_dct.abs(), target_dct.abs()) + self.loss(pred_dct.angle(), target_dct.angle()))

class dct_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        pred_dct = dct.batch_dct(pred)
        # print(pred.shape)
        # print(pred_dct.shape)
        target_dct = dct.batch_dct(target)

        return self.weight * (self.loss(pred_dct, target_dct))

class hfd_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, features, pred, target):
        b, _, h, w = pred.shape
        pred_dct = dct.batch_dct(pred)
        pred_dct = dct.zigzag(pred_dct)
        pred_dct = pred_dct.squeeze(1).transpose(1, 2).view(b, 64, 7, 7)

        target_dct = dct.batch_dct(target)
        target_dct = dct.zigzag(target_dct)
        target_dct = target_dct.squeeze(1).transpose(1, 2).view(b, 64, 7, 7)

        return self.weight * (self.loss(features, pred_dct - target_dct))

from torch.fft import fft as rfft
from torch.fft import ifft2
def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    fft = rfft(image, 2)
    fft_mag = torch.log(1 + torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2 + 1e-8))
    return fft_mag

class fft_loss():
    def __init__(self, weight=1.0):
        self.loss = torch.nn.L1Loss()
        self.weight = weight

    def __call__(self, pred, target):
        pred_fft = calc_fft(pred)
        target_fft = calc_fft(target)
        loss = self.loss(pred_fft, target_fft)

        return self.weight * loss


class MDFLoss(nn.Module):
    def __init__(self, saved_ds_path, cuda_available=True):
        super(MDFLoss, self).__init__()

        if cuda_available:
            self.Ds = torch.load(saved_ds_path)
        else:
            self.Ds = torch.load(saved_ds_path, map_location=torch.device('cpu'))

        self.num_discs = len(self.Ds)

    def forward(self, x, y, num_scales=9, is_ascending=1):
        # Get batch_size
        batch_size = x.shape[0]

        # Initialize loss vector
        loss = torch.zeros([batch_size]).to(x.device)
        # For every scale
        for scale_idx in range(num_scales):
            # Reverse if required
            if is_ascending:
                scale = scale_idx
            else:
                scale = self.num_discs - 1 - scale_idx

            # Choose discriminator
            D = self.Ds[scale]

            # Get discriminator activations
            pxs = D(x, is_loss=True)
            pys = D(y, is_loss=True)

            # For every layer in the output
            for idx in range(len(pxs)):
                # Compute L2 between representations
                l2 = (pxs[idx] - pys[idx]) ** 2
                l2 = torch.mean(l2, dim=(1, 2, 3))

                # Add current difference to the loss
                loss += l2

        # Mean loss
        loss = torch.mean(loss)

        return loss


class mdf_i_loss():
    def __init__(self, weight=1.0):
        self.loss = MDFLoss(saved_ds_path='./model/weights/Ds_i.pth')
        self.weight = weight

    def __call__(self, pred, target):
        loss = self.loss(pred, target)

        return self.weight * loss

class mdf_a_loss():
    def __init__(self, weight=1.0):
        self.loss = MDFLoss(saved_ds_path='./model/weights/Ds_a.pth')
        self.weight = weight

    def __call__(self, pred, target):
        loss = self.loss(pred, target)

        return self.weight * loss

class mdf_d_loss():
    def __init__(self, weight=1.0):
        self.loss = MDFLoss(saved_ds_path='./model/weights/Ds_d.pth')
        self.weight = weight

    def __call__(self, pred, target):
        loss = self.loss(pred, target)

        return self.weight * loss


###################################################################################################
############################## Embedding Consistency Loss #########################################
###################################################################################################
from train.model.submodules import *
import functools
class ec_loss():
    def __init__(self, weight=1.0):
        self.autoencoder = Autoencoder(norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True),
                          pool_layer=nn.MaxPool2d, leaky=True,
                          return_att=False, n1=16,
                          iters=1, upsampling='bilinear',
                          return_embedding=True)
        self.criterionL1 = torch.nn.L1Loss()
        self.weight = weight

    def __call__(self, pred_embedding, target, path):
        # print('loading the Autoencoder weights for loss from %s' % path)
        state_dict = torch.load(path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.cuda()
        # print('loading the Autoencoder weights for loss from %s succeed!' % path)
        _, self.autoencoder_embedding = self.autoencoder(target)

        return self.weight * self.criterionL1(self.autoencoder_embedding.detach(), pred_embedding)


####################################################################################################################
import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = True
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


class ff_loss():
    def __init__(self, weight=1.0):
        self.weight = weight
        self.loss = FocalFrequencyLoss(loss_weight=self.weight, alpha=1.0)

    def __call__(self, pred, target):
        loss = self.loss(pred, target)

        return loss


###################################################################################################
############################## Polarization Autoencoder Loss ######################################
###################################################################################################
from train.model.autoencoder import AutoEncoder
import functools
class pae_loss():
    def __init__(self, weight=1.0):
        self.autoencoder = AutoEncoder()
        self.criterionL1 = torch.nn.L1Loss()
        self.weight = weight

    def __call__(self, pred, target, path):
        state_dict = torch.load(path)
        self.autoencoder.load_state_dict(state_dict)
        self.autoencoder.cuda()
        with torch.no_grad():
            pred_e1, pred_e2, pred_e3, pred_e4, _ = self.autoencoder(pred)
            target_e1, target_e2, target_e3, target_e4, _ = self.autoencoder(target)

        # average in channel dimension
        pred_e1 = torch.mean(pred_e1, dim=1, keepdim=True)
        pred_e2 = torch.mean(pred_e2, dim=1, keepdim=True)
        pred_e3 = torch.mean(pred_e3, dim=1, keepdim=True)
        pred_e4 = torch.mean(pred_e4, dim=1, keepdim=True)
        target_e1 = torch.mean(target_e1, dim=1, keepdim=True)
        target_e2 = torch.mean(target_e2, dim=1, keepdim=True)
        target_e3 = torch.mean(target_e3, dim=1, keepdim=True)
        target_e4 = torch.mean(target_e4, dim=1, keepdim=True)

        loss = self.criterionL1(pred_e1, target_e1) + self.criterionL1(pred_e2, target_e2) + \
                   self.criterionL1(pred_e3, target_e3) + self.criterionL1(pred_e4, target_e4)
        # loss = 1 * self.criterionL1(pred_e1, target_e1) + 2 * self.criterionL1(pred_e2, target_e2) + \
        #        4 * self.criterionL1(pred_e3, target_e3) + 8 * self.criterionL1(pred_e4, target_e4)
        # loss = 8 * self.criterionL1(pred_e1, target_e1) + 4 * self.criterionL1(pred_e2, target_e2) + \
        #        2 * self.criterionL1(pred_e3, target_e3) + 1 * self.criterionL1(pred_e4, target_e4)

        return self.weight * loss
