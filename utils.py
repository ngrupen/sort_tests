import numpy as np
import cv2
from sklearn.utils.linear_assignment_ import linear_assignment

# --------------------------------------------------------------------------------------------------------------
# Object Detection
# --------------------------------------------------------------------------------------------------------------
class ObjectDetector():

    def __init__(self):
        # init motion detector -- can swap in any object detector here
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()

    def do_detection(self, detect_prob = 0.25):
        return int(np.random.choice(2, 1, p=[1 - detect_prob, detect_prob]))


    def detect(self, f, num):
        if self.do_detection() or num == 0:
        # if num % 5 == 0:
            # return bounding boxes for each detected object
            boxes = []

            # grayscale version of input frame, motion detection, find contours
            gray_f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            motion_mask = self.fgbg.apply(gray_f)
            contours, hierarchy = cv2.findContours(gray_f.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # construct list of bounding boxes from contours
            for c in contours:
                # avoid taking small artifacts
                if cv2.arcLength(c, True) > 100:
                    # prep for sort
                    (x, y, w, h) = cv2.boundingRect(c)
                    box = [x, y, x+w, y+h, None]
                    boxes.append(box)

            dets = np.asarray(boxes)
        else:
            # skip detection this iteration
            dets = []

        return dets



# --------------------------------------------------------------------------------------------------------------
# Functions for Data Association
# --------------------------------------------------------------------------------------------------------------
"""
As implemented in https://github.com/abewley/sort but with some modifications

For each detected item, it computes the intersection over union (IOU) w.r.t. each tracked object. (IOU matrix)

Then, it applies the Hungarian algorithm (via linear_assignment) to assign each det. item to the best possible
tracked item (i.e. to the one with max. IOU).
"""

def associate(dets, trks, iou_thresh=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if len(trks) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(dets)), np.empty((0,5),dtype=int)

    iou_matrix = np.zeros((len(dets),len(trks)),dtype=np.float32)


    for d,det in enumerate(dets):
        for t,trk in enumerate(trks):
            iou_matrix[d,t] = iou(det,trk)
 
    # Linear assignment minimizes total assignment cost.
    # Given iou matrix, we want to maximize total IOU between track predictions and frame detection
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d,det in enumerate(dets):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t,trk in enumerate(trks):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0],m[1]] < iou_thresh:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    print('matches = {}'.format(matches))
    print('unmatched detections = {}'.format(np.array(unmatched_detections)))
    print('unmatched trackers = {}'.format(np.array(unmatched_trackers)))

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def iou(bb_det, bb_trk):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_det[0], bb_trk[0])
    yy1 = np.maximum(bb_det[1], bb_trk[1])
    xx2 = np.minimum(bb_det[2], bb_trk[2])
    yy2 = np.minimum(bb_det[3], bb_trk[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h

    o = wh / ((bb_det[2]-bb_det[0])*(bb_det[3]-bb_det[1])
    + (bb_trk[2]-bb_trk[0])*(bb_trk[3]-bb_trk[1]) - wh)

    # print('iou between {} and {} = {}'.format(bb_det, bb_trk, o))

    return(o)




