import numpy as np
import torch
import os
import cv2
import math
import datetime
import os.path as osp
from scipy.spatial.distance import cdist
import time
from scipy.spatial.distance import cdist
import os.path as osp
from tqdm import tqdm

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/home/tl/KalmanMot/MOT16_Data/train'
label_root = '/home/tl/KalmanMot/MOT16_Data/generate_detection'
mkdirs(label_root)

seqs = [s for s in os.listdir(seq_root)]


for seq in tqdm(seqs):
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, visibility in tqdm(gt):
        if mark == 0 or not label == 1 :
            continue
        if visibility <= 0.005 :
            continue
        random_vis_delete_thres = np.random.randint(0,25)/100
        if visibility <= random_vis_delete_thres and np.random.random_sample() <= ( 1 - random_vis_delete_thres ) / 5 :
            continue
        random_permute = np.random.normal(0,0.01)
        random_permute = min(random_permute,0.01) if random_permute >0 else max(-0.01,random_permute)
        x,y,w,h = (1+random_permute)*x,(1+random_permute)*y,(1+random_permute)*w,(1+random_permute)*h
        fid = int(fid)
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
             x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)