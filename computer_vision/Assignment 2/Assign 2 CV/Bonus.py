import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_sift_matches(img1, img2, ratio_thresh=0.75, num_matches=50):
    """Detect SIFT keypoints/descriptors and return good matches."""
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector and detect keypoints & descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Use Brute-Force matcher with KNN (k=2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Sort matches by distance and pick best 'num_matches'
    good_matches = sorted(good_matches, key=lambda x: x.distance) # the lower the distance, the better the match
    if len(good_matches) > num_matches:
        good_matches = good_matches[:num_matches]

    return kp1, kp2, good_matches

def plot_matches(img1, kp1, img2, kp2, matches):
    """Plot the matched keypoints between two images."""
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("50 SIFT Correspondences")
    plt.axis("off")
    plt.show()

def compute_homography(kp_src, kp_dst, matches):
    """Compute homography matrix H mapping src → dst via RANSAC."""
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warp_image(src, H):
    """
    Warp src by H with clamped translation. Returns:
      - warped image on a canvas covering its full transformed extent
      - (t_x, t_y): the applied translation
    """
    h, w = src.shape[:2]
    # Corners of src
    corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)

    # Bounding box in float
    x_coords = warped_corners[:,:,0]
    y_coords = warped_corners[:,:,1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Integer bounds
    x_min_int = int(np.floor(x_min))
    y_min_int = int(np.floor(y_min))
    x_max_int = int(np.ceil(x_max))
    y_max_int = int(np.ceil(y_max))

    # Clamp translation to non-negative
    t_x = -x_min_int if x_min_int < 0 else 0
    t_y = -y_min_int if y_min_int < 0 else 0

    # Output canvas size
    width  = x_max_int - x_min_int
    height = y_max_int - y_min_int

    # Build translation matrix
    H_trans = np.array([
        [1, 0, t_x],
        [0, 1, t_y],
        [0, 0,   1]
    ])

    # Warp with inverse mapping + bilinear interpolation
    warped = cv2.warpPerspective(
        src,
        H_trans.dot(H),
        (width, height),
        flags=cv2.INTER_LINEAR
    )
    return warped, (t_x, t_y)

def create_mosaic(src, dst, H):
    """
    Warp src into dst's plane then overlay dst onto the combined canvas.
    """
    # 1. Warp src
    warped_src, (t_x, t_y) = warp_image(src, H)
    h_ws, w_ws = warped_src.shape[:2]
    h_dst, w_dst = dst.shape[:2]

    # 2. Prepare mosaic canvas
    W  = max(w_ws,  t_x + w_dst)
    Ht = max(h_ws, t_y + h_dst)
    mosaic = np.zeros((Ht, W, 3), dtype=np.uint8)

    # 3. Place warped src
    mosaic[:h_ws, :w_ws] = warped_src

    # 4. Build mask from dst (ensures matching shapes)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    mask = (dst_gray > 0)
    mask = np.dstack([mask, mask, mask])  # shape: (h_dst, w_dst, 3)

    # 5. Overlay dst
    y1, y2 = t_y, t_y + h_dst
    x1, x2 = t_x, t_x + w_dst
    region = mosaic[y1:y2, x1:x2]
    region[mask] = dst[mask]
    mosaic[y1:y2, x1:x2] = region

    return mosaic



if __name__ == "__main__":
    # Load images (order: left, middle, right)
    img1 = cv2.imread(r"C:\Users\ACER\Downloads\img1.jpg")
    img2 = cv2.imread(r"C:\Users\ACER\Downloads\img2.jpg")
    img3 = cv2.imread(r"C:\Users\ACER\Downloads\img3.jpg")

    # Check images loaded
    if img1 is None or img2 is None or img3 is None:
        raise IOError("Failed to load images. Check file paths.")

    # Step 1: Stitch left and middle images
    kp1, kp2, matches12 = get_sift_matches(img1, img2)
  #  plot_matches(img1, kp1, img2, kp2, matches12)
    H_12 = compute_homography(kp1, kp2, matches12)
    mosaic_12 = create_mosaic(img1, img2, H_12)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(mosaic_12, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("mosaic_12")
    plt.show()

    # Step 2: Stitch mosaic with right image
    kp3, kp_mosaic, matches_32 = get_sift_matches(img3, mosaic_12)
   # plot_matches(img3, kp3, mosaic_12, kp_mosaic, matches_32)
    H_32 = compute_homography(kp3, kp_mosaic, matches_32)
    Final_Panorama = create_mosaic(img3, mosaic_12, H_32)

    # Display final mosaic
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(Final_Panorama, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Final_Panorama")
    plt.show()