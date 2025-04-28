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


def compute_homography(kp1, kp2, matches):
    """Compute homography matrix using matched keypoints."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask


def warp_image(src, H):
    """
    Warp an image using a given homography matrix via inverse mapping.

    The function computes the output canvas size based on where the
    input image corners map under the homography, applies a translation to
    ensure all coordinates are positive, and then warps the image.
    """
    h, w = src.shape[:2]
    # Define source image corners
    corners = np.float32([[0, 0],
                          [w - 1, 0],
                          [w - 1, h - 1],
                          [0, h - 1]]).reshape(-1, 1, 2)

    # Warp the corners to determine output dimensions
    warped_corners = cv2.perspectiveTransform(corners, H)

    # Get bounding box [xmin, ymin, xmax, ymax]
    [xmin, ymin] = np.int32(warped_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(warped_corners.max(axis=0).ravel() + 0.5)

    # Translation to shift the image so that all pixels are in positive coordinates
    t = [-xmin, -ymin]
    H_translation = np.array([[1, 0, t[0]],
                              [0, 1, t[1]],
                              [0, 0, 1]])

    # Warp the source image using the combined homography
    warped_img = cv2.warpPerspective(src, H_translation.dot(H), (xmax - xmin, ymax - ymin))
    return warped_img, H_translation, t


def create_mosaic(img1, img2, H):
    """
    Creates a mosaic by warping the first image into the coordinate system of
    the second image and merging both views onto a single canvas.
    """
    # Warp img1 using the homography H
    warped_img1, H_translation, t = warp_image(img1, H)

    # Determine mosaic canvas size
    h_warp, w_warp = warped_img1.shape[:2]
    h_img2, w_img2 = img2.shape[:2]

    # Place img2 on the mosaic canvas using the translation offset computed earlier.
    t_x, t_y = t
    mosaic_width = max(w_warp, t_x + w_img2)
    mosaic_height = max(h_warp, t_y + h_img2)

    # Initialize mosaic canvas (black background)
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    # Place the warped image into the mosaic canvas
    mosaic[0:h_warp, 0:w_warp] = warped_img1

    # Overlay the second image: create region where img2 will be placed
    mosaic_region = mosaic[t_y:t_y + h_img2, t_x:t_x + w_img2]

    # Create mask from img2 where the pixel intensity is non-zero.
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Create a boolean mask (h, w) and then stack into (h, w, 3)
    mask = (img2_gray > 0)
    mask = np.dstack([mask, mask, mask])

    # Overlay img2 onto the mosaic region using the mask
    mosaic_region[mask] = img2[mask]
    mosaic[t_y:t_y + h_img2, t_x:t_x + w_img2] = mosaic_region

    return mosaic


if __name__ == "__main__":
    # Load the two images (update file paths as required)
    img1 = cv2.imread("D:\lectures\Term_10\Assignment2\part2_first.jpeg")
    img2 = cv2.imread("D:\lectures\Term_10\Assignment2\part2_second.jpeg")

    if img1 is None or img2 is None:
        raise IOError("Please check the file paths for the input images.")

    # 1. Getting Correspondences using SIFT and BFMatcher with KNN & Ratio Test
    kp1, kp2, good_matches = get_sift_matches(img1, img2, ratio_thresh=0.75, num_matches=50)
    plot_matches(img1, kp1, img2, kp2, good_matches)

    # 2. Compute the Homography from the correspondences (using RANSAC)
    H, mask = compute_homography(kp1, kp2, good_matches)
    print("Computed Homography Matrix:")
    print(H)

    # 3. Warp the first image into the plane of the second image
    warped_img1, H_translation, t = warp_image(img1, H)
    """plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image 1")
    plt.axis("off")
    plt.show()
"""
    # 4. Create the final mosaic by merging the warped image with the second image
    mosaic = create_mosaic(img1, img2, H)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    plt.title("Final Mosaic")
    plt.axis("off")
    plt.show()

    # Optionally, save the mosaic
    cv2.imwrite("mosaic_output.jpg", mosaic)