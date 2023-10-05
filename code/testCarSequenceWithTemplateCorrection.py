import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

frames = np.load("../data/carseq.npy")
org_rect = [59, 116, 145, 151]
rect_p = org_rect.copy()

frame_0 = frames[:,:,0].copy()
template = frame_0.copy()
rects = []
p_n  = np.zeros(2, dtype=np.float64)
p_prev = np.zeros(2, dtype=np.float64)
for i in range(frames.shape[2]-1):
    frame_t1 = frames[:,:,i+1].copy()
    p = LucasKanade(template, frame_t1, rect_p, threshold, num_iters, p_prev)
    p_n = np.array([rect_p[0] - org_rect[0] + p[0], rect_p[1] - org_rect[1] + p[1]])
    pstar = LucasKanade(frame_0, frame_t1, org_rect, threshold, num_iters, p_n)
    
    if (np.linalg.norm(pstar - p_n, ord = 1) <= template_threshold):
        template = frames[:,:,i+1].copy()
        rect_p = [org_rect[0] + pstar[0], org_rect[1] + pstar[1], org_rect[2] + pstar[0], org_rect[3] + pstar[1]]
        p_prev = np.array([p_n[0] - pstar[0], p_n[1] - pstar[1]])
    else:
        p_prev = np.array([org_rect[0] + p_n[0] - rect_p[0], org_rect[1] + p_n[1] - rect_p[1]])

    out_rect = [org_rect[0] + pstar[0], org_rect[1] + pstar[1], org_rect[2] + pstar[0], org_rect[3] + pstar[1]]
    rects.append(out_rect)

np.save('carseqrects-wcrt.npy', rects)
