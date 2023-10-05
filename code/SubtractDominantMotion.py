import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.interpolate import RectBivariateSpline
import cv2

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    # Input:
    # 	Images at time t and t+1
    #  Output:
    # 	mask: [nxm]
    #  put your implementation here
    
    mask = np.zeros(image1.shape, dtype=bool)

    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # use inverse composition
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    M = np.vstack((M, np.array([[0, 0, 1]])))
    M = np.linalg.inv(M)

    interpolator_1 = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    interpolator_2 = RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)

    x = np.arange(0, image1.shape[1])
    y = np.arange(0, image1.shape[0])
    X, Y = np.meshgrid(x, y)
    X_warped = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
    Y_warped = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
    
    invalid = (X_warped < 0) | (X_warped >= image1.shape[1]) | (Y_warped < 0) & (Y_warped >= image1.shape[0])
    interped_I1 = interpolator_1.ev(Y_warped, X_warped)
    interped_I2 = interpolator_2.ev(Y, X)
    interped_I1[invalid] = 0
    interped_I2[invalid] = 0

    # calculate the difference
    diff = abs(interped_I2 - interped_I1)
    ind = (diff > tolerance) & (interped_I2 != 0)
    mask[ind] = 1
    mask = binary_dilation(mask)

    return mask