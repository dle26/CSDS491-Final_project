from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('--task', default='mot', help='mot')
    self.parser.add_argument('--dataset', default='jde', help='jde')
    # tracking
    self.parser.add_argument('--track_buffer', type=int, default = 30, help='tracking buffer')
    self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    self.parser.add_argument('--input_path', type=str, default='../MOT16_Data/train/MOT16-02/img1', help='path to the input video')
    self.parser.add_argument('--input_detection_path', type=str, default='../MOT16_Data/generate_detection/MOT16-02', help='path to the input video')
    self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
    self.parser.add_argument('--use_cam_motion', type=bool, default = False, help='using camera motion or not')
    self.parser.add_argument('--use_dynamic_retrack', type=bool, default = False, help='using dynamic retracking or not')
    self.parser.add_argument('--use_reranking',  default= False, help='')
    self.parser.add_argument('--output-root', type=str, default='../results/MOT16-02', help='expected output root path')
    self.parser.add_argument('--use_hog_reid', default = False, help='Use hog for reid')
    self.parser.add_argument('--use_kalman', default =  False, help='Use hog for reid')
    self.parser.add_argument('--use_iou', default = True, help='Use hog for reid')
    # mot
  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
    return opt
  def init(self, args=''):
    
    opt = self.parse(args)
    return opt
  