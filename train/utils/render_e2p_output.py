import cv2
import numpy as np
import torch

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
    dolp = torch2cv2(output['d'])
    a_chans=output['a']
    if len(a_chans.shape)==4:
        a_sin=a_chans[:,[0],...]
        a_cos=a_chans[:,[1],...]
        aolp_01=torch.atan2(a_sin,a_cos)
    else: # gt gives us aolp directly
        aolp_01=a_chans
    aolp=torch2cv2(aolp_01)
    # aolp = torch2cv2(output['a']) # the DNN aolp output 0 correspond to polarization angle -pi/2 and 1 correspond to +pi/2
    # find the DoLP values that are less than mask value and use them to mask out the AoLP values so they show up as black
    aolp_mask=np.where(dolp<dolp_aolp_mask_level*255) #2d array with 1 where aolp is valid, 0 otherwise
    aolp_mask_torch=output['d'].ge(dolp_aolp_mask_level)
    # compute median angle in radians, with zero being *vertical* polarization
    median_aolp_angle=compute_median_aolp(aolp_01,aolp_mask_torch).cpu().numpy().item()

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
    aolp_hsv = cv2.applyColorMap(aolp, cv2.COLORMAP_HSV)

    # mask out AoLP pixels with low DoLP
    aolp_hsv[aolp_mask]=0


    # draw the median AoLP angle on top of HSV AoLP image
    if not np.isnan(median_aolp_angle):
        line_thickness = 2
        half_line_length=50
        line_color=(200, 200, 200)
        sin,cos=np.sin(median_aolp_angle),np.cos(median_aolp_angle)
        (x,y)=(int(half_line_length*sin),int(half_line_length*cos)) # note sin is x since vertical is 0 angle
        (h,w)=aolp.shape
        cv2.line(aolp_hsv, (int(w/2-x),int(h/2-y)), (int(w/2+x),int(h/2+y)), line_color, thickness=line_thickness)

    # show DoLP using hot code (black to white with red/yellow on the way)
    dolp = cv2.applyColorMap(dolp, cv2.COLORMAP_HOT)
    return intensity, aolp_hsv, dolp

def compute_median_aolp(aolp_output, aolp_mask):
    ''' computes the median angle in radians from the 0-1 range aolp_output image
    :param aolp_output: the DNN torch float output image (2d)
    :param mask: a binary mask that is True where we want to compute median, False to ignore
    :returns: the scalar median angle in radians with pi/2 and -pi/2 horizontal, 0 vertical polarization
    '''
    unmasked_aolp_values=torch.masked_select(aolp_output,aolp_mask)
    median_angle=np.pi*(torch.median(unmasked_aolp_values)-0.5)
    return median_angle
