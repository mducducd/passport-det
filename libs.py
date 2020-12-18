import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt 
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import timeit
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, get_coor
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np