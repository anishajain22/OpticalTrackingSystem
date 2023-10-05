import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    # Input:
    # 	It: template image
    # 	It1: Current image
    #  Output:
    # 	M: the Affine warp matrix [2x3 numpy array]
    #   put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()
    delta_p = np.array([It1.shape[1]]*6)

    interpolator2 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    x = np.arange(0, It.shape[1])
    y = np.arange(0, It.shape[0])
    
    for _ in range(int(num_iters)):
        
        X, Y = np.meshgrid(x, y)
        X_warped = p[0]*X + p[1]*Y + p[2]
        Y_warped = p[3]*X + p[4]*Y + p[5]
        # find valid positions
        valid = (X_warped > 0) & (X_warped < It.shape[1]) & (Y_warped > 0) & (Y_warped < It.shape[0])
        X_warped = X_warped[valid]
        Y_warped= Y_warped[valid]
        warped_It1 = interpolator2.ev(Y_warped, X_warped)

        # calculate gradients
        gradient_x = interpolator2.ev(Y_warped, X_warped, dx=0, dy=1).flatten()
        gradient_y = interpolator2.ev(Y_warped, X_warped, dx=1, dy=0).flatten()

        # get matrix A
        X = X[valid].flatten()
        Y = Y[valid].flatten()

        A = np.zeros((gradient_x.shape[0], 6))
        X, Y = X.flatten(), Y.flatten()
        A[:, 0] = np.multiply(gradient_x, X)
        A[:, 1] = np.multiply(gradient_x, Y)
        A[:, 2] = gradient_x
        A[:, 3] = np.multiply(gradient_y, X)
        A[:, 4] = np.multiply(gradient_y, Y)
        A[:, 5] = gradient_y

        # get matrix b
        b = It[valid].flatten() - warped_It1.flatten()

        delta_p = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), b))

        p += delta_p.flatten()
        if np.sum(delta_p ** 2) <= threshold:
            break

    M = np.reshape(p, (2, 3))

    return M