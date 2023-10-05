import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import cv2

# write your script here, we recommend all or some of the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq',
    default='../data/aerialseq.npy',
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file_path = args.seq

seq = np.load(seq_file_path)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''

masks = []
frames_of_interest = {30, 60, 90, 120}
for i in range(seq.shape[2]-1):
    print('frame %d'%i)
    frame_t = seq[:,:,i].copy()
    frame_t1 = seq[:,:,i+1].copy()
    if i in frames_of_interest:
        mask = SubtractDominantMotion(frame_t, frame_t1, threshold, num_iters, tolerance)

        frame_t1 = (frame_t1 * 255).astype(np.uint8)

        frame_t1_color = cv2.cvtColor(frame_t1, cv2.COLOR_GRAY2BGR)

        frame_t1_color[mask] = [0, 0, 255]

    # if i in frames_of_interest:
        plt.imshow(frame_t1_color)
        plt.title('frame %d'%i)
        plt.savefig('aerialmaskframe_%d.png'%i)
