import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    # Input:
    # 	It: template image
    # 	It1: Current image
    #  Output:
    # 	M: the Affine warp matrix [2x3 numpy array]
    #   put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()
    delta_p = np.array([It1.shape[1]]*6)

    interpolator1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    interpolator2 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    X, Y = np.meshgrid(x, y)

    gradient_x = interpolator1.ev(Y, X, dx=0, dy=1).flatten()
    gradient_y = interpolator1.ev(Y, X, dx=1, dy=0).flatten()

    A = np.zeros((gradient_x.shape[0], 6))
    X_flatten, Y_flatten = X.flatten(), Y.flatten()
    A[:, 0] = np.multiply(gradient_x, X_flatten)
    A[:, 1] = np.multiply(gradient_x, Y_flatten)
    A[:, 2] = gradient_x
    A[:, 3] = np.multiply(gradient_y, X_flatten)
    A[:, 4] = np.multiply(gradient_y, Y_flatten)
    A[:, 5] = gradient_y
    
    for _ in range(int(num_iters)):
        X_warped = p[0]*X + p[1]*Y + p[2]
        Y_warped = p[3]*X + p[4]*Y + p[5]
        valid = (X_warped > 0) & (X_warped < It.shape[1]) & (Y_warped > 0) & (Y_warped < It.shape[0])
        X_warped = X_warped[valid]
        Y_warped = Y_warped[valid]
        warped_It1 = interpolator2.ev(Y_warped, X_warped)
        
        A_valid = A[valid.flatten()]
        b =  - It[valid].flatten() + warped_It1.flatten()

        delta_p = np.dot(np.linalg.inv(np.dot(np.transpose(A_valid), A_valid)), np.dot(np.transpose(A_valid), b))

        delta_M = np.vstack((np.reshape(delta_p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M[0, 0] += 1
        delta_M[1, 1] += 1
        M = np.dot(M, np.linalg.inv(delta_M))
        p = M[:2, :].flatten()
        if np.sum(delta_p ** 2) <= threshold:
            break

    return M[:2,:]