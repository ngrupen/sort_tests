import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import time 
from collections import OrderedDict
from scipy.spatial import distance as dist
from utils import *
from sort import Sort

def do_detection(detect_prob=0.25):
    return int(np.random.choice(2, 1, p=[1 - detect_prob, detect_prob]))

def main(fp):
    # init detector, sort 
    detector = ObjectDetector()
    tracker = Sort()

    # load video
    cap = cv2.VideoCapture(fp)

    time = 0.0
    frame_count = 0

    while True:
        # read next frame and resize
        ret, f = cap.read()
        f = cv2.resize(f, (0, 0), None, .5, .5)

        # detections for current frame
        dets = detector.detect(f, frame_count)

        # update object detections -- returns dict of current objects
        objs = tracker.update(dets)

        for o in objs:
            # each object is [start_x, start_y, end_x, end_y, ID]
            cv2.rectangle(f, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (0, 255, 0), 2)
            cv2.putText(f, str(int(o[4])), (int(o[0]) + 10, int(o[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        frame_count += 1
        
        cv2.imshow('Frame', f)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy optical flow')
    parser.add_argument('--dir', default='../data/flow_vids/single_move.mp4', type=str, help='video directory')
    args = parser.parse_args()
    main(args.dir)