from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
from cv2 import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from tracker.multitracker import JDETracker
from tracker.basetrack import TrackState
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.data_loader as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_ids, tlwhs, track_ids in results:
            for frame_id, tlwh, track_id in zip(frame_ids, tlwhs, track_ids):
                if data_type == 'kitti':
                    frame_id -= 1
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None,bbox_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    if bbox_dir:
        mkdir_if_missing(bbox_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0

    for path, img0, detection in dataloader:
        if frame_id % 1 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
                       

        # run tracking
        timer.tic()
        online_targets,detection_boxes = tracker.update(detection, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        #bbox detection plot        
        box_tlbrs=[]
        box_scores=[]
        img_bbox=img0.copy()
        for box in detection_boxes:
            tlbr=box.tlbr
            tlwh=box.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                box_tlbrs.append(tlbr)
                box_scores.append(box.score)

        timer.toc()
        # save results
        results.append( ( [frame_id + 1]*len(online_tlwhs), online_tlwhs, online_ids) )
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
            bbox_im=vis.plot_detections(img_bbox,box_tlbrs,scores=box_scores)
        if show_image:
            cv2.imshow('online_im', online_im)
            cv2.imshow('bbox_im',bbox_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
            cv2.imwrite(os.path.join(bbox_dir, '{:05d}.jpg'.format(frame_id)), bbox_im)

        frame_id += 1
    # save results
    track_pools = []
    id_pools = []
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


