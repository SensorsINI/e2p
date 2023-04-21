import cv2
import numpy as np
from train.utils import torch2cv2

def render_e2p_output(output, dolp_aolp_mask_level, brightness):
    """ Computes intensity, aolp, dolp output color images from  E2P output.
    :param output: DNN output
    :param dolp_aolp_mask_level: values of AoLP are set to zero when the DoLP is less than this value (0-1)
    :param brightness: modified brightness of intensity channel, nominally 1.0, multiplies the intensity channel
    :returns: intensity, aolp, dolp RGB color 2d images that are uint8 from 0-255, suitable for cv2.imshow rendering
        the DNN aolp output 0 correspond to polarization angle -pi/2 and 255 correspond to +pi/2
    """
    intensity = torch2cv2(output['i'])*brightness
    # compute aolp mod 1 to unwrap the wrapped AoLP
    # aolp = torch2cv2(output['a']%1) # the DNN aolp output 0 correspond to polarization angle -pi/2 and 1 correspond to +pi/2
    aolp = torch2cv2(output['a']) # the DNN aolp output 0 correspond to polarization angle -pi/2 and 1 correspond to +pi/2
    dolp = torch2cv2(output['d'])
    aolp_mask=np.where(dolp<dolp_aolp_mask_level*255)

    # find the DoLP values that are less than mask value and use them to mask out the AoLP values so they show up as black


    # #visualizing aolp, dolp on tensorboard, tensorboard takes rgb values in [0,1]
    # this makes visualization work correctly in tensorboard.
    # a lot of ugly formatting things to make cv2 output compatible with matplotlib.
    # first scale to [0,255] then convert to colormap_hsv,
    # then swap axis to make bgr to rgb otherwise the color is reversed.
    # then last convert the rgb color back to [0,1] for tensorboard....
    # imgs.append(torch.Tensor(cv2.cvtColor(cv2.applyColorMap(np.moveaxis(np.asarray(pred_aolp.cpu()*255).astype(np.uint8), 0, -1), cv2.COLORMAP_HSV), cv2.COLOR_BGR2RGB)).permute(2,0,1).cuda()/255.)
    # imgs.append(torch.Tensor(cv2.cvtColor(cv2.applyColorMap(np.moveaxis(np.asarray(pred_dolp.cpu()*255).astype(np.uint8), 0, -1), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)).permute(2,0,1).cuda()/255.)

    # https://stackoverflow.com/questions/50963283/imshow-doesnt-need-convert-from-bgr-to-rgb
    # no need to convert since we use cv2 to render to screen
    # intensity = norm_max_min(intensity)
    intensity = np.repeat(intensity[:, :, None], 3, axis=2).astype(np.uint8) # need to duplicate mono channels to make compatible with aolp and dolp
    aolp = cv2.applyColorMap(aolp, cv2.COLORMAP_HSV)
    aolp[aolp_mask]=0
    dolp = cv2.applyColorMap(dolp, cv2.COLORMAP_HOT)
    return intensity, aolp, dolp
