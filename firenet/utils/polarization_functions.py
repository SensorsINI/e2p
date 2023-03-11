"""
 @Time    : 18.05.22 12:37
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : polarization_functions.py
 @Function:
 
"""
import os
import cv2
import math
import numpy as np


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def ExtractRGGB(image_result_data):
    h, w = image_result_data.shape[:2]

    row_idx = np.asarray([[i, i + 1] for i in range(0, w, 4)]).flatten()
    col_idx = np.asarray([[i, i + 1] for i in range(0, h, 4)]).flatten()
    R = image_result_data[col_idx][:, row_idx]

    row_idx = np.asarray([[i, i + 1] for i in range(2, w, 4)]).flatten()
    col_idx = np.asarray([[i, i + 1] for i in range(0, h, 4)]).flatten()
    G1 = image_result_data[col_idx][:, row_idx]

    row_idx = np.asarray([[i, i + 1] for i in range(0, w, 4)]).flatten()
    col_idx = np.asarray([[i, i + 1] for i in range(2, h, 4)]).flatten()
    G2 = image_result_data[col_idx][:, row_idx]

    row_idx = np.asarray([[i, i + 1] for i in range(2, w, 4)]).flatten()
    col_idx = np.asarray([[i, i + 1] for i in range(2, h, 4)]).flatten()
    B = image_result_data[col_idx][:, row_idx]

    return R, G1, G2, B


def ExtractPolarization(S):
    h, w = S.shape[:2]

    IM90 = S[0::2, 0::2]
    IM45 = S[0::2, 1::2]
    IM135 = S[1::2, 0::2]
    IM0 = S[1::2, 1::2]

    return IM0, IM45, IM90, IM135


def CalcADoLPnpy(im0_16, im45_16, im90_16, im135_16):
    im_stokes0 = im0_16.astype(np.float64) + im90_16.astype(np.float64)
    im_stokes1 = im0_16.astype(np.float64) - im90_16.astype(np.float64)
    im_stokes2 = im45_16.astype(np.float64) - im135_16.astype(np.float64)

    im_DOLP = np.divide(
        np.sqrt(np.square(im_stokes1) + np.square(im_stokes2)),
        im_stokes0,
        out=np.zeros_like(im_stokes0), where=im_stokes0 != 0.0
    ).astype(np.float64)

    im_DOLP = im_DOLP.clip(0.0, 1.0)

    # normalize from [0.0, 1.0] range to [0, 255] range (8 bit)
    im_DOLP_normalized = (im_DOLP * 255).astype(np.uint8)

    # create a heatmap from the greyscale image
    im_DOLP_heatmap = cv2.applyColorMap(im_DOLP_normalized, cv2.COLORMAP_HOT)

    im_AOLP = (0.5 * np.arctan2(im_stokes2, im_stokes1)).astype(np.float64)
    im_AOLP[im_stokes2 < 0] += math.pi
    # print('%s aolp range:[%f, %f]' % (cname, np.min(im_AOLP), np.max(im_AOLP)))

    # normalize from [-pi/2, pi/2] range to [0, 255] range (8 bit)
    im_AOLP_normalized = (im_AOLP*(255/math.pi)).astype(np.uint8)

    # create a heatmap from the greyscale image
    im_AOLP_heatmap = cv2.applyColorMap(im_AOLP_normalized, cv2.COLORMAP_JET)

    # return im_AOLP_heatmap, im_DOLP_normalized
    return im_AOLP_normalized, im_AOLP_heatmap, im_DOLP_normalized, im_DOLP_heatmap


def RecoverColor(R, G1, G2, B):
    h, w = R.shape[:2]

    r = R[0::2, 0::2].astype(np.int32) + R[1::2, 1::2].astype(np.int32)
    g1 = G1[0::2, 0::2].astype(np.int32) + G1[1::2, 1::2].astype(np.int32)
    g2 = G2[0::2, 0::2].astype(np.int32) + G2[1::2, 1::2].astype(np.int32)
    b = B[0::2, 0::2].astype(np.int32) + B[1::2, 1::2].astype(np.int32)
    g = ((g1 + g2) / 2)

    r = Normalization(r)
    g = Normalization(g)
    b = Normalization(b)

    return cv2.merge([b, g, r])


def Normalization(S):
    S = (S.astype(np.float) / 2.0).astype(np.uint16)

    return S


def CalcADoLP(image_result_data):
    R, G1, G2, B = ExtractRGGB(image_result_data)

    IM0_R, IM45_R, IM90_R, IM135_R = ExtractPolarization(R)
    aolp_r, aolp_rc, dolp_r, dolp_rc = CalcADoLPnpy(IM0_R, IM45_R, IM90_R, IM135_R)

    IM0_G, IM45_G, IM90_G, IM135_G = ExtractPolarization(G1)
    aolp_g, aolp_gc, dolp_g, dolp_gc = CalcADoLPnpy(IM0_G, IM45_G, IM90_G, IM135_G)

    IM0_B, IM45_B, IM90_B, IM135_B = ExtractPolarization(B)
    aolp_b, aolp_bc, dolp_b, dolp_bc = CalcADoLPnpy(IM0_B, IM45_B, IM90_B, IM135_B)

    color = RecoverColor(R, G1, G2, B)

    return [aolp_r, aolp_g, aolp_b], [aolp_rc, aolp_gc, aolp_bc], [dolp_r, dolp_g, dolp_b], [dolp_rc, dolp_gc, dolp_bc], color
