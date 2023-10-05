import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import time

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2, dtype=np.float64)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    height, width = It.shape
    p = p0
    x1, y1, x2, y2 = rect
    
    x, y = np.meshgrid(np.arange(x1, x2+1), np.arange(y1, y2+1), indexing='ij')
    x = x.flatten()
    y = y.flatten()

    interpolator1 = RectBivariateSpline(np.arange(height), np.arange(width), It)
    interpolator2 = RectBivariateSpline(np.arange(height), np.arange(width), It1)

    error = None
    for _ in np.arange(num_iters):
        
        x_warped = x + p[0]
        y_warped = y + p[1]
        
        valid_indices = (x_warped >= 0) & (x_warped < width) & (y_warped >= 0) & (y_warped < height)
        
        It1_warped = interpolator2.ev(y_warped, x_warped)

        if valid_indices.any():
            It_val = interpolator1.ev(y, x)
            error = It_val - It1_warped
        else:
            if error is None:
                error = np.zeros_like(It1_warped)

        gradient_x = interpolator2.ev(y_warped, x_warped, dx=0, dy=1)
        gradient_y = interpolator2.ev(y_warped, x_warped, dx=1, dy=0)
        
        A = np.vstack((gradient_x.flatten(), gradient_y.flatten())).T

        delta_p = np.linalg.lstsq(A, error, rcond=None)[0]
        p += delta_p
        
        if np.linalg.norm(delta_p)**2 < threshold:
            break

    return p
