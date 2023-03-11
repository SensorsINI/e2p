"""
 @Time    : 06.10.22 10:30
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : firenet-pdavis
 @File    : model_v.py
 @Function:
 
"""
import math

import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.base_model import BaseModel
from .submodules_v import *

from .legacy import FireNet_legacy
