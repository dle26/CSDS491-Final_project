from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.data_loader as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadImages(opt.input_path, opt.input_detection_path) #, out_size = (640,480)
    result_filename = os.path.join(result_root, 'baseline.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    bbox_dir  = None if opt.output_format == 'text' else osp.join(result_root, 'bbox_detection')
    eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir,bbox_dir=bbox_dir, show_image=False, frame_rate=frame_rate)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)
def run_mot_16(opt):
    list_vid=['MOT16-02'] #,'MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13'
    for vid in list_vid:
        result_root = '../results/' + vid
        mkdir_if_missing(result_root)

        logger.info('Starting tracking...')
        out_size = (1920,1080) if vid !='MOT16-05' else (640,480)
        dataloader = datasets.LoadImages('../MOT16_Data/train/' +  vid + '/img1', '../MOT16_Data/generate_detection/' +  vid ,
                                         out_size = out_size)
        result_filename = os.path.join(result_root, 'iou.txt')
        frame_rate = dataloader.frame_rate

        frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
        bbox_dir  = None if opt.output_format == 'text' else osp.join(result_root, 'bbox_detection')
        eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir,bbox_dir=bbox_dir, show_image=False, frame_rate=frame_rate)

        if opt.output_format == 'video':
            output_video_path = osp.join(result_root, 'result.mp4')
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
            os.system(cmd_str)

if __name__ == '__main__':
    opt = opts().init()
    # demo(opt)
    run_mot_16(opt)
