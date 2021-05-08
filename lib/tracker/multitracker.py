from collections import deque
import  copy
import numpy as np

from tracker import matching
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from .basetrack import BaseTrack, TrackState
from scipy.spatial.distance import cdist
import cv2
from PIL import Image

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    out_of_frame_patience=5
    num_cluster=5
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0


        self.smooth_feat = None

       
        self.update_features(temp_feat,None)
        self.alpha = 0.9
        self.occlusion_status=False # use for bbox only
        self.iou_box=None #use for bbox only
        self.num_out_frame=0
        
            
        
    def update_features(self, feat, new_track):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        
       
    @staticmethod
    def warp_predict(mean, cov, warp_matrix, warp_mode):
        if warp_matrix is None:
            return mean, cov
        track_xyah = mean[:4]
        track_tlwh = STrack.xyah_to_tlwh(track_xyah)
        track_tlbr = STrack.tlwh_to_tlbr(track_tlwh)
        t,l,b,r = track_tlbr
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_tlbr  = cv2.perspectiveTransform(np.array([[[t,l],[b,r]]]), warp_matrix)[0].flatten()
        else:
            warp_tlbr  = cv2.transform(np.array([[[t,l],[b,r]]]), warp_matrix)[0].flatten()
       
        warp_tlwh  = STrack.tlbr_to_tlwh(warp_tlbr)
        warp_xyah  = STrack.tlwh_to_xyah(warp_tlwh)
        track_mean, track_cov = list(warp_xyah) + list(mean[4:]) , cov
        return np.array(track_mean), track_cov
    @staticmethod
    def get_camera_intension(warp_matrix,warp_mode):
        if warp_matrix is None:
            return 0
        warp_matrix_flattern = warp_matrix.flatten()
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            non_change_warp = np.array([1,0,0,0,1,0,0,0,1])
        else:
            non_change_warp = np.array([1,0,0,0,1,0]) 
        similarity = np.dot(warp_matrix_flattern,non_change_warp)/(np.sqrt(np.sum(warp_matrix_flattern**2)) * np.sqrt(np.sum(non_change_warp**2)))
        return 1 - similarity 

    def predict(self, warp_matrix,warp_mode, smooth = 0.0):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        motion_intensity = STrack.get_camera_intension(warp_matrix,warp_mode) * smooth
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, motion_intensity = motion_intensity )
        self.mean, self.covariance = STrack.warp_predict(self.mean, self.covariance, warp_matrix,warp_mode)

    @staticmethod
    def multi_predict(stracks, warp_matrix,warp_mode, smooth = 0.0):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            motion_intensity = STrack.get_camera_intension(warp_matrix,warp_mode) * smooth
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, motion_intensity = motion_intensity)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                stracks[i].mean, stracks[i].covariance = STrack.warp_predict(stracks[i].mean, stracks[i].covariance, warp_matrix,warp_mode)
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        # self.box_hist.append(self.tlwh)
        # self.track_frames.append(frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat,new_track)
        #self.update_cluster(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

       
        

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat,new_track)
        

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def xyah_to_tlwh(xyah):
        w,h = xyah[2]*xyah[3], xyah[3]
        x,y = xyah[0],xyah[1]
        t,l = x-w/2, y-h/2
        return [t,l,w,h]

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.use_hog_reid:
            print('USE HOG AS FEATURE EXTRACTION !!!!')
            self.re_im_h,self.re_im_w=300,120
            cell_size = (50, 24)  
            block_size = (6, 5)
            nbins = 9  
            self.reid_model = cv2.HOGDescriptor(_winSize=(self.re_im_w // cell_size[1] * cell_size[1],
                                  self.re_im_h // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        

        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.prev_img = None        

        self.kalman_filter = KalmanFilter()
        


    @staticmethod
    def get_warp_matrix (prev_img, img0, warp_mode, resize_factor = 1):
        size = (prev_img.shape[1]//resize_factor), (prev_img.shape[0]//resize_factor)
        resize_prev_img = cv2.resize(prev_img,size) if resize_factor !=1 else prev_img
        resize_img0 = cv2.resize(img0,size) if resize_factor !=1 else img0
        number_of_iterations = 25
        termination_eps = 1e-10
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps )
        _,warp_matrix = cv2.findTransformECC(cv2.cvtColor(resize_prev_img,cv2.COLOR_BGR2GRAY), cv2.cvtColor(resize_img0,cv2.COLOR_BGR2GRAY),
                                             warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix[0][2] = warp_matrix[0][2]*resize_factor
            warp_matrix[1][2] = warp_matrix[1][2]*resize_factor
            warp_matrix[2][0] = warp_matrix[2][0]/resize_factor
            warp_matrix[2][1] = warp_matrix[2][1]/resize_factor
        else :
            warp_matrix[0][2] = warp_matrix[0][2]*resize_factor
            warp_matrix[1][2] = warp_matrix[1][2]*resize_factor
        
        return warp_matrix


    def update(self, detection, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        
        ''' Step 1: Network forward, get detections & embeddings'''
        dets = []
        for x,y,w,h in detection :
            t,l,b,r = x -w/2, y-h/2, x + w/2, y + h/2
            dets.append([t*width,l*height,b*width,r*height,1])
        dets = np.array(dets)
        
        id_feature=[]
        if self.opt.use_hog_reid :
            for box in dets[:, :5]:
                try:
                    x1,y1,x2,y2,conf=max(int(box[0]),0),max(int(box[1]),0),min(int(box[2]),width-1),min(int(box[3]),height-1),box[4]
                    id_feature.append(self.reid_model.compute(cv2.resize(img0[y1:y2,x1:x2:,],(self.re_im_w,self.re_im_h)))[:,0])
                except:
                    id_feature.append(np.zeros_like(id_feature[-1]))
        else:
            id_feature =  np.zeros((len(dets),1))

        warp_mode = cv2.MOTION_TRANSLATION
        if self.prev_img is not None and self.opt.use_cam_motion == True:
            warp_matrix = self.get_warp_matrix(self.prev_img, img0.copy(), warp_mode , resize_factor = 4)
        else:
            warp_matrix = None
        if self.opt.use_cam_motion:
            self.prev_img = img0


        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []
        
        detections_plot=detections.copy()
        

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool,lost_map_tracks = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
            
        STrack.multi_predict(strack_pool, warp_matrix, warp_mode)

        if self.opt.use_hog_reid :
            dists = matching.embedding_distance(strack_pool, detections) if not self.opt.use_reranking else matching.reranking_embeding_distance(strack_pool, detections)
        else :
            dists=np.zeros(shape=(len(strack_pool),len(detections)))
        
            
        if self.opt.use_kalman :
            dists = matching.fuse_motion(self.opt,self.kalman_filter, dists, strack_pool, detections,lost_map=lost_map_tracks,lambda_ = 0.99)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh = 0.7) #0.6

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
        else:
            u_detection = range(len(detections))
            u_track = range(len(strack_pool))
        r_stracks = strack_pool
        ''' Step 3: Second association, with IOU'''
        if self.opt.use_iou :
            detections = [detections[i] for i in u_detection]
            if self.opt.use_kalman:
                r_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            else:
                r_stracks = [strack_pool[i] for i in u_track ]

            dists = matching.iou_distance(r_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9)  #0.7

            for itracked, idet in matches:
                track = r_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
        ''' '''
        for it in u_track:
            track = r_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            

       
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            cam_veloc_weight = 0.95
          
            if self.opt.use_dynamic_retrack :
                cam_motion = STrack.get_camera_intension(warp_matrix, warp_mode) 
                track_vtlwh = np.array(STrack.xyah_to_tlwh(track.mean[4:]))
                track_vtlbr = STrack.tlwh_to_tlbr(track_vtlwh)
                veloc_motion = np.sqrt(np.sum(track_vtlbr**2)) 
                max_time_lost = self.max_time_lost * 3.2 * np.exp(-(cam_veloc_weight * cam_motion + (1 - cam_veloc_weight) * veloc_motion))
            else:
                max_time_lost = self.max_time_lost
            if self.frame_id - track.end_frame > max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
            #Remove out of screen tracklet
            elif track.tlwh[0]+track.tlwh[2]//2>width or track.tlwh[1]+track.tlwh[3]//2>height:
                track.num_out_frame+=1
                if track.num_out_frame>STrack.out_of_frame_patience:
                    track.mark_removed()
                    removed_stracks.append(track)

        # print('Remained match {} s'.format(t4-t3))
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks,_ = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks,_ = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        #merge track
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        print('===========Frame {}=========='.format(self.frame_id))
        print('Activated: {}'.format([track.track_id for track in activated_starcks]))
        print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        print('Lost: {}'.format([track.track_id for track in self.lost_stracks]))
        print('Removed: {}'.format([track.track_id for track in self.removed_stracks]))
        
        return output_stracks,detections_plot




def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    lost_map=[]
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
        lost_map.append(0)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
            lost_map.append(1)
    return res,lost_map


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
