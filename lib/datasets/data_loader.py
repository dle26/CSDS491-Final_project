import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import copy
from opts import opts



class LoadImages:  # for inference
    def __init__(self, img_folder, detection_folder, out_size=(1920, 1080)):
            
        image_format = ['.jpg', '.jpeg', '.png', '.tif']
        self.files = sorted(glob.glob('%s/*.*' % img_folder))
        self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))

        self.detection_folder = detection_folder
        self.nF = len(self.files)  # number of image files
        self.w = out_size[0]
        self.h = out_size[1]
        self.count = 0
        seq_info = open(osp.join(img_folder[:-5], 'seqinfo.ini')).read()
        self.frame_rate = int(seq_info[seq_info.find('frameRate=') + 10:seq_info.find('\nseqLength')])

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]
        detect_file = self.detection_folder + '/' +img_path.split('/')[-1].split('.')[0] + '.txt'
        img_detection = np.loadtxt(detect_file, dtype=np.float64, delimiter=' ')

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path
        img0 = cv2.resize(img0, (self.w, self.h))
    
        return img_path, img0, img_detection

    

    def __len__(self):
        return self.nF  # number of files


