import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in format [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
             + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

class Track:
    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,0,1,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.x[:4] = np.array([[bbox[0]],[bbox[1]],[bbox[2]],[bbox[3]]])
        self.time_since_update = 0
        self.id = track_id
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.bbox = bbox

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array(bbox))
        self.bbox = bbox

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - numpy array of detections in format [[x1,y1,x2,y2,score],[...],...]
        Returns:
          A similar array with tracked objects [[x1,y1,x2,y2,track_id],...]
        """
        # Predict new locations
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.tracks):
            pos = trk.predict()
            trks[t, :4] = pos
            trks[t, 4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        # Associate detections to tracked objects
        if len(dets) > 0 and len(trks) > 0:
            iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
            for d, det in enumerate(dets):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = iou(det, trk)
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(matched_indices).T

            unmatched_dets = []
            for d in range(len(dets)):
                if d not in matched_indices[:,0]:
                    unmatched_dets.append(d)

            unmatched_trks = []
            for t in range(len(trks)):
                if t not in matched_indices[:,1]:
                    unmatched_trks.append(t)

            # Update matched tracks
            for m in matched_indices:
                self.tracks[m[1]].update(dets[m[0], :4])

        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(trks)))

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            trk = Track(dets[i,:4], self.track_id_count)
            self.track_id_count += 1
            self.tracks.append(trk)

        # Remove dead tracks
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            if trk.time_since_update > self.max_age:
                self.tracks.pop(i-1)
            i -= 1

        # Return track list
        for trk in self.tracks:
            if (trk.hits >= self.min_hits) or (self.track_id_count <= self.min_hits):
                ret.append(np.concatenate((trk.bbox, [trk.id])).reshape(1,-1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))
