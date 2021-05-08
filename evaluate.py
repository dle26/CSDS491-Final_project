from lib.tracking_utils.evaluation import Evaluator
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import logging
import os
import os.path as osp
import motmetrics as mm
import numpy as np

#run twostage)demo to get result file, then run this
#file to get evaluate result
#'MOT16-02','MOT16-04','MOT16-09','MOT16-10','MOT16-11','MOT16-13',
#'MOT16-05','MOT16-10','MOT16-11','MOT16-13'
def main( data_root='../results', seqs=('MOT16-02',), exp_name='demo'):
    logger.setLevel(logging.INFO)
    
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
       
        result_root = os.path.join(data_root,seq)
        result_filename = os.path.join(result_root, '{}.txt'.format('kalman_iou'))
        print(result_filename)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        
  

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

main()