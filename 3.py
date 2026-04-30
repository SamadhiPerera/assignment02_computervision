import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load Images
# =========================
img1 = cv2.imread('c1.jpg')  # Circuit board c1
img2 = cv2.imread('c2.jpg')  # Circuit board c2

# Convert to RGB for plotting
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# =========================
# (a) MANUAL HOMOGRAPHY
# =========================
# Select 4 corresponding points manually
# Format: [x, y]

# Points in image 1 (source)
pts1 = np.float32([
    [100, 100],
    [500, 120],
    [520, 800],
    [120, 780]
])

# Corresponding points in image 2 (destination)
pts2 = np.float32([
    [150, 150],
    [520, 180],
    [540, 820],
    [180, 800]
])

# Compute homography
H, _ = cv2.findHomography(pts1, pts2)

# Warp image1 to perspective of image2
warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

# Display warped image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image 1")
plt.imshow(img1_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Warped Image 1 -> Image 2 Perspective")
plt.imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# =========================
# (b) DIFFERENCE IMAGE
# =========================
# Convert to grayscale
gray_warped = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Absolute difference
diff = cv2.absdiff(gray_img2, gray_warped)

plt.figure(figsize=(6, 6))
plt.title("Difference Image (Manual Homography)")
plt.imshow(diff, cmap='gray')
plt.axis('off')
plt.show()


# =========================
# (c) SIFT FEATURE MATCHING
# =========================
# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher
bf = cv2.BFMatcher()

# KNN matching
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test (Lowe's test)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
match_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12, 6))
plt.title("SIFT Feature Matches")
plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# =========================
# (d) HOMOGRAPHY USING MATCHES
# =========================
# Extract matched points
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography using RANSAC
H_sift, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp image using SIFT homography
warped_sift = cv2.warpPerspective(img1, H_sift, (img2.shape[1], img2.shape[0]))

# Show warped image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Warped (SIFT Homography)")
plt.imshow(cv2.cvtColor(warped_sift, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reference Image (Image 2)")
plt.imshow(img2_rgb)
plt.axis('off')
plt.show()


# =========================
# DIFFERENCE IMAGE (SIFT)
# =========================
gray_warped_sift = cv2.cvtColor(warped_sift, cv2.COLOR_BGR2GRAY)

diff_sift = cv2.absdiff(gray_img2, gray_warped_sift)

plt.figure(figsize=(6, 6))
plt.title("Difference Image (SIFT Homography)")
plt.imshow(diff_sift, cmap='gray')
plt.axis('off')
plt.show()