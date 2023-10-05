import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# write your script here, we recommend the above libraries for making your animation

from LucasKanade import LucasKanade
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
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

frames = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
print(frames.shape)
height = rect[3] - rect[1]
width = rect[2] - rect[0]
print(height, width)
rects = []

for i in range(frames.shape[2]-1):
    frame_t = frames[:,:,i].copy()
    frame_t1 = frames[:,:,i+1].copy()
    p = LucasKanade(frame_t, frame_t1, rect, threshold, num_iters)
    rect = [rect[0] + p[0], rect[1] + p[1], rect[2] + p[0], rect[3] + p[1]]
    rects.append(rect)

np.save('carseqrects.npy', rects)

  