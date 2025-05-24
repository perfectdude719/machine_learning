import cv2
import numpy as np

def dynamic_programming_stereo(left_img, right_img, sigma=2, c0=1):
    # Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    if len(right_img.shape) == 3:
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    h, w = left_img.shape
    sigma_sq = sigma ** 2
    disparity = np.zeros((h, w), dtype=np.float32)
    
    for row in range(h):
        I_l = left_img[row, :].astype(np.float32)
        I_r = right_img[row, :].astype(np.float32)
        W = len(I_l)
        D = np.full((W, W), np.inf, dtype=np.float32)
        D[0, 0] = (I_l[0] - I_r[0]) ** 2 / sigma_sq
        
        # Fill DP matrix
        for i in range(W):
            for j in range(W):
                if i == 0 and j == 0:
                    continue
                
                candidates = []
                if i > 0 and j > 0:
                    d_ij = (I_l[i] - I_r[j]) ** 2 / sigma_sq
                    candidates.append(D[i-1, j-1] + d_ij)
                if i > 0:
                    candidates.append(D[i-1, j] + c0)
                if j > 0:
                    candidates.append(D[i, j-1] + c0)
                
                if candidates:
                    D[i, j] = min(candidates)
        
        # Backtracking
        i, j = W-1, W-1
        row_disp = np.zeros(W, dtype=np.float32)
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and D[i, j] == D[i-1, j-1] + ((I_l[i] - I_r[j]) ** 2)/sigma_sq:
                row_disp[i] = abs(i - j)
                i -= 1
                j -= 1
            elif i > 0 and D[i, j] == D[i-1, j] + c0:
                row_disp[i] = 0
                i -= 1
            else:
                j -= 1
        
        disparity[row, :] = row_disp
    
    # Normalize for display
    return cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def opencv_disparity(left_img, right_img):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=11,
        P1=8*3*11**2,
        P2=32*3*11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = stereo.compute(left_img, right_img)
    return cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Example usage
left = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# Compute disparities
custom_disp = dynamic_programming_stereo(left, right)
opencv_disp = opencv_disparity(left, right)

# Display results
cv2.imshow('Custom DP Disparity', custom_disp)
cv2.imshow('OpenCV SGBM Disparity', opencv_disp)
cv2.waitKey(0)
cv2.destroyAllWindows()