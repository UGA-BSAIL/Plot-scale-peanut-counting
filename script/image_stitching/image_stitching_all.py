import cv2
import numpy as np
import os
import re

import pandas as pd
from pathlib import Path
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures


def load_images_from_folder(folder, width_roi_start=0.2, width_roi_end=0.8):
    """
    Load images from a specified folder, crop them based on the region of interest (ROI),
    and return the cropped images in a list.

    Args:
        folder (str): The folder path containing the image files.
        width_roi_start (float): The starting fraction of the width for the ROI. Defaults to 0.2 (20%).
        width_roi_end (float): The ending fraction of the width for the ROI. Defaults to 0.8 (80%).

    Returns:
        list: A list of cropped images.
    """
    # Get a list of all files in the folder
    files = os.listdir(folder)
    # Use a regular expression to extract numbers from filenames and sort them in descending order
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()), reverse=True)

    images = []
    for filename in files:
        print(filename)  # Print the current filename being processed
        img = cv2.imread(os.path.join(folder, filename))  # Read the image from the file path

        if img is not None:
            # Get the dimensions of the image (height, width, and channels)
            height, width, _ = img.shape

            # Calculate the starting and ending pixel positions for the width cropping
            crop_start = int(width * width_roi_start)
            crop_end = int(width * width_roi_end)

            # Crop the image based on the calculated positions
            cropped_img = img[:, crop_start:crop_end]

            # Append the cropped image to the list
            images.append(cropped_img)

    # Return the list of cropped images
    return images


def detect_and_describe(image, method='sift'):
    """
    Detects and describes features in an image using various feature detection algorithms.
    
    Args:
        image (numpy.ndarray): Input image.
        method (str): Feature detection and description method to use. Defaults to 'sift'.

    Returns:
        keypoints (list): Detected keypoints.
        descriptors (numpy.ndarray): Descriptors corresponding to the keypoints.
    """
    # Check the selected method and apply the corresponding feature detection algorithm
    if method == 'sift':
        # SIFT (Scale-Invariant Feature Transform)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

    elif method == 'surf':
        # SURF (Speeded-Up Robust Features)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image, None)

    elif method == 'orb':
        # ORB (Oriented FAST and Rotated BRIEF)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

    elif method == 'brisk':
        # BRISK (Binary Robust Invariant Scalable Keypoints)
        brisk = cv2.BRISK_create()
        keypoints, descriptors = brisk.detectAndCompute(image, None)

    elif method == 'akaze':
        # AKAZE (Accelerated-KAZE)
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(image, None)

    elif method == 'kaze':
        # KAZE (Nonlinear Diffusion-Based Features)
        kaze = cv2.KAZE_create()
        keypoints, descriptors = kaze.detectAndCompute(image, None)

    elif method == 'daisy':
        # DAISY (Dense Keypoint Description)
        daisy = cv2.xfeatures2d.DAISY_create()
        keypoints = daisy.detect(image)  # Detect keypoints
        keypoints, descriptors = daisy.compute(image, keypoints)  # Compute descriptors

    elif method == 'lucid':
        # LUCID (Locally Uniform Comparison Image Descriptor)
        lucid = cv2.xfeatures2d.LUCID_create()
        keypoints = lucid.detect(image)
        keypoints, descriptors = lucid.compute(image, keypoints)

    elif method == 'latch':
        # LATCH (Learned Arrangements of Three Patch Codes)
        latch = cv2.xfeatures2d.LATCH_create()
        keypoints = latch.detect(image)
        keypoints, descriptors = latch.compute(image, keypoints)

    elif method == 'freak':
        # FREAK (Fast Retina Keypoint)
        freak = cv2.xfeatures2d.FREAK_create()
        keypoints = freak.detect(image)
        keypoints, descriptors = freak.compute(image, keypoints)

    elif method == 'superpoint':
        # SuperPoint (Deep Learning-Based Keypoint Detector)
        superpoint = cv2.superres.DenseOpticalFlow_ext_createSuperPoint()
        keypoints = superpoint.detect(image)
        keypoints, descriptors = superpoint.compute(image, keypoints)

    elif method == 'd2net':
        # D2-Net (Deep Learning-Based Keypoint Descriptor)
        d2net = cv2.dnn.readNetFromTorch('d2net/d2_tf.pth')  # Load D2-Net model
        keypoints = d2net.detect(image)
        keypoints, descriptors = d2net.compute(image, keypoints)

    # GPU-accelerated methods
    elif method == 'sift_gpu':  # SIFT using GPU
        sift = cv2.cuda.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)  # Upload image to GPU
        keypoints = sift.detect(gray_gpu, None)
        keypoints, descriptors = sift.compute(gray_gpu, keypoints)

    elif method == 'surf_gpu':  # SURF using GPU
        surf = cv2.cuda.SURF_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = surf.detect(gray_gpu, None)
        keypoints, descriptors = surf.compute(gray_gpu, keypoints)

    elif method == 'orb_gpu':  # ORB using GPU
        orb = cv2.cuda_ORB.create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = orb.detect(gray_gpu, None)
        keypoints, descriptors = orb.compute(gray_gpu, keypoints)

    elif method == 'fast_gpu':  # FAST using GPU
        fast = cv2.cuda.FastFeatureDetector_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = fast.detect(gray_gpu, None)
        keypoints, descriptors = fast.compute(gray_gpu, keypoints)

    elif method == 'brisk_gpu':  # BRISK using GPU
        brisk = cv2.cuda.createBRISK()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = brisk.detect(gray_gpu, None)
        keypoints, descriptors = brisk.compute(gray_gpu, keypoints)

    elif method == 'akaze_gpu':  # AKAZE using GPU
        akaze = cv2.cuda.createAKAZE()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = akaze.detect(gray_gpu, None)
        keypoints, descriptors = akaze.compute(gray_gpu, keypoints)

    elif method == 'kaze_gpu':  # KAZE using GPU
        kaze = cv2.cuda.createKAZE()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = kaze.detect(gray_gpu, None)
        keypoints, descriptors = kaze.compute(gray_gpu, keypoints)

    elif method == 'superpoint_gpu':  # SuperPoint using GPU
        superpoint = cv2.cuda_SuperPoint.create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = superpoint.detect(gray_gpu, None)
        keypoints, descriptors = superpoint.compute(gray_gpu, keypoints)

    elif method == 'd2net_gpu':  # D2-Net using GPU
        d2net = cv2.cuda_D2Net.create('d2net/d2_tf.pth')  # Load D2-Net model for GPU
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = d2net.detect(gray_gpu, None)
        keypoints, descriptors = d2net.compute(gray_gpu, keypoints)

    else:
        # Raise an error for unsupported feature detection methods
        raise ValueError(f"Unknown feature detection method: {method}")


def match_features(kp1, desc1, kp2, desc2, lowe_ratio=0.75, cross_check=True):
    """
    Matches features between two sets of keypoints and descriptors using FLANN-based matching.

    Args:
        kp1 (list): Keypoints from the first image.
        desc1 (numpy.ndarray): Descriptors for keypoints in the first image.
        kp2 (list): Keypoints from the second image.
        desc2 (numpy.ndarray): Descriptors for keypoints in the second image.
        lowe_ratio (float): Lowe's ratio for filtering matches. Defaults to 0.75.
        cross_check (bool): If True, applies cross-checking to verify matches. Defaults to True.

    Returns:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        good_matches (list): List of good matches after applying filtering criteria.
    """
    # Define parameters for FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN index for KD-Tree algorithm
    search_params = dict(checks=50)  # Number of checks for finding nearest neighbors

    # Initialize FLANN-based matcher
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Ensure descriptors are in float32 format (required by FLANN)
    desc1 = np.float32(desc1)
    desc2 = np.float32(desc2)

    # Perform k-nearest neighbors matching (k=2)
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        # Retain match if the distance of the best match is significantly smaller
        # than the distance of the second-best match
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    # Apply cross-checking if specified
    if cross_check:
        good_matches_cc = []
        # Perform matching in the reverse direction
        matches_reverse = matcher.knnMatch(desc2, desc1, k=1)
        for m in good_matches:
            # Check if the reverse match points back to the original keypoint
            if matches_reverse[m.trainIdx][0].trainIdx == m.queryIdx:
                good_matches_cc.append(m)
        # Update matches with cross-checked results
        good_matches = good_matches_cc

    # Return the keypoints and filtered matches
    return kp1, kp2, good_matches


def find_homography(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.99):
    """
    Computes a homography matrix between two sets of keypoints using RANSAC.

    Args:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): List of good matches between the two sets of keypoints.
        max_reproj_error (float): Maximum allowed reprojection error for RANSAC. Defaults to 3.0.
        confidence (float): Confidence level for the estimated homography. Defaults to 0.99.

    Returns:
        H (numpy.ndarray): Homography matrix if successfully computed, otherwise None.
        status (numpy.ndarray): Array indicating inliers and outliers (1 for inliers, 0 for outliers).
    """
    # Ensure there are enough matches to compute a homography
    if len(matches) >= 4:  # At least 4 matches are required
        # Extract point coordinates from the matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])  # Points in the first image
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])  # Points in the second image

        # Compute the homography matrix using RANSAC
        # RANSAC helps to robustly estimate the homography by filtering outliers
        H, status = cv2.findHomography(
            points1, points2, cv2.RANSAC,
            ransacReprojThreshold=max_reproj_error,
            confidence=confidence
        )

        # Return the homography matrix and the RANSAC inlier/outlier status
        return H, status
    else:
        # If there are fewer than 4 matches, return None
        return None, None


def find_similarity_transformation(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.99):
    """
    Estimates a similarity transformation matrix between two sets of keypoints using RANSAC.

    Args:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): List of good matches between the two sets of keypoints.
        max_reproj_error (float): Maximum allowed reprojection error for RANSAC. Defaults to 3.0.
        confidence (float): Confidence level for the estimated transformation. Defaults to 0.99.

    Returns:
        H (numpy.ndarray): A 3x3 similarity transformation matrix if successfully computed, otherwise None.
        status (numpy.ndarray): Array indicating inliers and outliers (1 for inliers, 0 for outliers).
    """
    # Ensure there are enough matches to compute the transformation
    if len(matches) >= 4:  # At least 4 matches are required
        # Extract point coordinates from the matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])  # Points in the first image
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])  # Points in the second image

        # Estimate the affine transformation matrix using RANSAC
        # The affine transformation is a partial 2D transformation (rotation, scaling, translation)
        M, status = cv2.estimateAffinePartial2D(
            points1, points2, method=cv2.RANSAC,
            ransacReprojThreshold=max_reproj_error,
            confidence=confidence
        )

        # Convert the affine matrix to a 3x3 format compatible with homography-based functions
        if M is not None:
            # Append a row [0, 0, 1] to make the matrix a 3x3 transformation
            H = np.vstack([M, [0, 0, 1]])
            return H, status
        else:
            # Return None if the transformation matrix could not be estimated
            return None, None
    else:
        # Return None if there are insufficient matches
        return None, None

    
def find_similarity_transformation_noscale(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.99):
    """
    Estimates a similarity transformation matrix (excluding scaling) between two sets of keypoints using RANSAC.

    Args:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): List of good matches between the two sets of keypoints.
        max_reproj_error (float): Maximum allowed reprojection error for RANSAC. Defaults to 3.0.
        confidence (float): Confidence level for the estimated transformation. Defaults to 0.99.

    Returns:
        H (numpy.ndarray): A 3x3 similarity transformation matrix (without scaling) if successfully computed, otherwise None.
        status (numpy.ndarray): Array indicating inliers and outliers (1 for inliers, 0 for outliers).
    """
    # Ensure there are enough matches to compute the transformation
    if len(matches) >= 4:  # At least 4 matches are required
        # Extract point coordinates from the matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])  # Points in the first image
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])  # Points in the second image

        # Estimate the affine transformation matrix using RANSAC
        # The affine transformation includes rotation, scaling, and translation
        M, status = cv2.estimateAffinePartial2D(
            points1, points2, method=cv2.RANSAC,
            ransacReprojThreshold=max_reproj_error,
            confidence=confidence
        )

        if M is not None:
            # Extract the rotation and scaling component
            T = M[:, 2]  # Translation vector (last column)
            R = M[:, 0:2]  # Rotation and scaling matrix (first two columns)

            # Normalize the rotation matrix to remove scaling
            # Compute the determinant of R and take the square root to get the scale factor
            norm = np.sqrt(np.linalg.det(R))

            # Divide R by the scale factor to normalize it
            R /= norm

            # Recombine the normalized rotation and translation components
            M = np.hstack([R, T.reshape(-1, 1)])  # Combine R and T into a 2x3 matrix

            # Convert to a 3x3 homography matrix format by appending [0, 0, 1]
            H = np.vstack([M, [0, 0, 1]])

            # Return the homography matrix (excluding scaling) and the RANSAC inlier mask
            return H, status
        else:
            # Return None if the transformation matrix could not be estimated
            return None, None
    else:
        # Return None if there are insufficient matches
        return None, None

    
def find_translation_transformation(kp1, kp2, matches, primary_axis='y'):
    """
    Estimates a translation transformation matrix along a primary axis (x or y).

    Args:
        kp1 (list): Keypoints from the first image.
        kp2 (list): Keypoints from the second image.
        matches (list): List of good matches between the two sets of keypoints.
        primary_axis (str): The axis along which to compute translation ('x' or 'y'). Defaults to 'y'.

    Returns:
        T (numpy.ndarray): A 3x3 translation matrix.
        None: Placeholder for status, as this function does not use inlier/outlier distinction.
    """
    # Ensure there are enough matches to compute translation
    if len(matches) >= 2:  # At least 2 matches are required
        # Extract the coordinates of matched keypoints
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])  # Keypoints in the first image
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])  # Keypoints in the second image

        # Compute the translation differences along the primary axis
        if primary_axis == 'x':
            # Compute translations along the x-axis
            translations = points2[:, 0] - points1[:, 0]
        else:
            # Compute translations along the y-axis (default)
            translations = points2[:, 1] - points1[:, 1]

        # Use the median of the translations to minimize the effect of outliers
        main_translation = np.median(translations)

        # Construct the 3x3 translation matrix
        if primary_axis == 'x':
            # Translation along the x-axis
            T = np.array([[1, 0, main_translation],  # Translate in x
                          [0, 1, 0],                # No translation in y
                          [0, 0, 1]])               # Homogeneous coordinates
        else:
            # Translation along the y-axis
            T = np.array([[1, 0, 0],                # No translation in x
                          [0, 1, main_translation],  # Translate in y
                          [0, 0, 1]])               # Homogeneous coordinates

        # Return the translation matrix and a placeholder for the status
        return T, None
    else:
        # Return None if there are insufficient matches
        return None, None


def display_matches(matched_result, image_index, max_width=1280, max_height=2400):
    """
    Displays an image of matched features between two images, resizing it if necessary to fit within specified limits.

    Args:
        matched_result (numpy.ndarray): The image showing the matched features.
        image_index (int): The index of the current image for display purposes.
        max_width (int): Maximum allowable width for the displayed image. Defaults to 1280.
        max_height (int): Maximum allowable height for the displayed image. Defaults to 2400.
    """
    # Get the original height and width of the image
    h, w = matched_result.shape[:2]

    # Calculate the aspect ratio of the image
    aspect_ratio = w / h

    # Determine the new dimensions based on the maximum size constraints
    if h > max_height or w > max_width:
        # Check if scaling by height or width is more restrictive
        if max_height / h < max_width / w:
            # Scale based on the height constraint
            new_height = max_height
            new_width = int(new_height * aspect_ratio)  # Maintain the aspect ratio
        else:
            # Scale based on the width constraint
            new_width = max_width
            new_height = int(new_width / aspect_ratio)  # Maintain the aspect ratio
    else:
        # Use the original dimensions if within size limits
        new_width, new_height = w, h

    # Resize the image to the new dimensions
    resized_image = cv2.resize(matched_result, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Display the resized image
    # Uncomment the lines below to display the image in a window
    # cv2.imshow(f'Matched features between image {image_index - 1} and image {image_index}', resized_image)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the display window


def drawc_matches(img1, img2, keypoints1, keypoints2):
    """
    Draws matched keypoints between two images and connects them with lines for visualization.

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        keypoints1 (list): List of keypoints in the first image (each keypoint is a tuple of coordinates).
        keypoints2 (list): List of keypoints in the second image (each keypoint is a tuple of coordinates).

    Returns:
        combined_img (numpy.ndarray): The combined image with matches visualized.
    """
    # Ensure both images have the same height for side-by-side concatenation
    if img1.shape[0] != img2.shape[0]:
        target_height = max(img1.shape[0], img2.shape[0])  # Determine the target height
        # Resize img1 to the target height while maintaining the aspect ratio
        img1 = cv2.resize(img1, (int(img1.shape[1] * target_height / img1.shape[0]), target_height))
        # Resize img2 to the target height while maintaining the aspect ratio
        img2 = cv2.resize(img2, (int(img2.shape[1] * target_height / img2.shape[0]), target_height))

    # Ensure img1 is in color (3 channels)
    if len(img1.shape) == 2:  # If img1 is grayscale (1 channel)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)  # Convert to color
    else:
        img1_color = img1  # Already in color

    # Ensure img2 is in color (3 channels)
    if len(img2.shape) == 2:  # If img2 is grayscale (1 channel)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)  # Convert to color
    else:
        img2_color = img2  # Already in color

    # Concatenate the two images side-by-side
    combined_img = np.concatenate((img1_color, img2_color), axis=1)

    # Compute the horizontal offset for the second image in the combined view
    offset = img1.shape[1]

    # Draw the matches by iterating through keypoint pairs
    for pt1, pt2 in zip(keypoints1, keypoints2):
        # Extract coordinates of the keypoint in the first image
        x1, y1 = int(pt1[0]), int(pt1[1])
        # Extract coordinates of the keypoint in the second image and apply the offset
        x2, y2 = int(pt2[0] + offset), int(pt2[1])

        # Draw a red circle at the keypoint location in the first image
        cv2.circle(combined_img, (x1, y1), 5, (0, 0, 255), -1)  # Red dot for img1 keypoints
        # Draw a red circle at the keypoint location in the second image
        cv2.circle(combined_img, (x2, y2), 5, (0, 0, 255), -1)  # Red dot for img2 keypoints

        # Draw a blue line connecting the matched keypoints
        cv2.line(combined_img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue line for matches

    # Return the combined image with matches visualized
    return combined_img



# ----------calculate the Homography matrix----------
def calculate_H_with_loftr_y_translation(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    """
    Calculates a homography matrix that represents translation along the Y-axis between two images using LoFTR.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.

    Returns:
        H (numpy.ndarray): A 3x3 homography matrix representing translation along the Y-axis.
        None: If not enough matches are found.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors and move them to the device (GPU/CPU)
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.  # Normalize to [0, 1]
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.  # Normalize to [0, 1]

    # Initialize LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image as the query
        "image1": img2_gray,  # Second image as the reference
    }

    # Perform feature matching with LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on the confidence threshold
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Ensure there are enough matches for RANSAC to work
    if len(mkpts0) > 4:
        # Calculate the differences in Y-coordinates between matched keypoints
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]

        # Prepare data for RANSAC regression
        X = mkpts0[:, 1].reshape(-1, 1)  # Y-coordinates of keypoints in the first image (independent variable)
        y = y_diff.reshape(-1, 1)        # Y-coordinate differences (dependent variable)

        # Initialize and fit RANSAC regressor
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(X, y)

        # Extract inlier mask from RANSAC results
        inlier_mask = ransac.inlier_mask_

        # Calculate the mean translation along the Y-axis using inliers
        ty = np.mean(y_diff[inlier_mask])  # Compute the mean translation from inliers

        # Construct the homography matrix for translation along the Y-axis
        H = np.array([
            [1, 0, 0],  # No translation along X
            [0, 1, ty],  # Translation along Y
            [0, 0, 1]    # Homogeneous coordinates
        ])
        return H
    else:
        # Print an error message if not enough matches are found
        print('Not enough matches were found.')
        return None


def calculate_H_with_loftr_xy_translation(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    """
    Calculates a homography matrix that represents translation along both X and Y axes 
    between two images using LoFTR and RANSAC.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.

    Returns:
        H (numpy.ndarray): A 3x3 homography matrix representing translation along X and Y axes.
        None: If not enough matches are found.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors and normalize to [0, 1], then move them to the device
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize the LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare the input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image (query)
        "image1": img2_gray,  # Second image (reference)
    }

    # Perform feature matching using LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on confidence threshold
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Ensure there are enough matches for RANSAC to work
    if len(mkpts0) > 4:
        # Calculate differences in X and Y coordinates
        x_diff = mkpts1[:, 0] - mkpts0[:, 0]  # Differences in X coordinates
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]  # Differences in Y coordinates

        # Prepare data for RANSAC regression
        X = mkpts0  # Independent variables: coordinates from the first image
        y = np.vstack([x_diff, y_diff]).T  # Dependent variables: differences in X and Y

        # Initialize and fit RANSAC for robust translation estimation
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(X, y)

        # Extract the inlier mask from RANSAC
        inlier_mask = ransac.inlier_mask_

        # Calculate mean translations along X and Y using inliers
        tx = np.mean(x_diff[inlier_mask])  # Mean translation along X-axis
        ty = np.mean(y_diff[inlier_mask])  # Mean translation along Y-axis

        # Construct the homography matrix for translation along X and Y axes
        H = np.array([
            [1, 0, tx],  # Translation in X
            [0, 1, ty],  # Translation in Y
            [0, 0, 1]    # Homogeneous coordinates
        ])
        return H
    else:
        # Print an error message if there are insufficient matches
        print('Not enough matches were found.')
        return None


def calculate_H_with_loftr_xy_rotation_translation(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    """
    Calculates a homography matrix that includes translation along X and Y axes and rotation
    between two images using LoFTR and RANSAC.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.

    Returns:
        H (numpy.ndarray): A 3x3 homography matrix representing rotation and translations.
        None: If not enough matches are found.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors, normalize to [0, 1], and move them to the device
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize the LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare the input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image (query)
        "image1": img2_gray,  # Second image (reference)
    }

    # Perform feature matching using LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on confidence threshold
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Ensure there are enough matches for RANSAC to work
    if len(mkpts0) > 4:
        # Calculate translations in X and Y
        x_diff = mkpts1[:, 0] - mkpts0[:, 0]  # Differences in X coordinates
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]  # Differences in Y coordinates

        # Calculate angles to estimate rotation
        angles0 = np.arctan2(mkpts0[:, 1], mkpts0[:, 0])  # Angles of keypoints in the first image
        angles1 = np.arctan2(mkpts1[:, 1], mkpts1[:, 0])  # Angles of keypoints in the second image
        angle_diffs = angles1 - angles0  # Differences in angles (rotation)

        # Prepare data for RANSAC regression
        X = mkpts0  # Independent variables: coordinates from the first image
        y = np.vstack([x_diff, y_diff, angle_diffs]).T  # Dependent variables: translations and rotation

        # Initialize and fit RANSAC for robust parameter estimation
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(X, y)

        # Extract the inlier mask from RANSAC
        inlier_mask = ransac.inlier_mask_

        # Calculate mean translations and rotation angle using inliers
        tx = np.mean(x_diff[inlier_mask])  # Mean translation along X-axis
        ty = np.mean(y_diff[inlier_mask])  # Mean translation along Y-axis
        rotation = np.mean(angle_diffs[inlier_mask])  # Mean rotation angle in radians

        # Construct the homography matrix for rotation and translations
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)

        H = np.array([
            [cos_theta, -sin_theta, tx],  # Rotation and X translation
            [sin_theta,  cos_theta, ty],  # Rotation and Y translation
            [0, 0, 1]                    # Homogeneous coordinates
        ])
        return H
    else:
        # Print an error message if there are insufficient matches
        print('Not enough matches were found.')
        return None


def calculate_H_with_loftr_xy_scale_translation(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    """
    Calculates a homography matrix that includes translation along X and Y axes and scaling
    between two images using LoFTR and RANSAC.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.

    Returns:
        H (numpy.ndarray): A 3x3 homography matrix representing scaling and translations.
        None: If not enough matches are found.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors, normalize to [0, 1], and move them to the device
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize the LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare the input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image (query)
        "image1": img2_gray,  # Second image (reference)
    }

    # Perform feature matching using LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on confidence threshold
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Ensure there are enough matches for RANSAC to work
    if len(mkpts0) > 4:
        # Calculate differences in X and Y coordinates
        x_diff = mkpts1[:, 0] - mkpts0[:, 0]  # Differences in X coordinates
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]  # Differences in Y coordinates

        # Calculate distances between matched keypoints to estimate scaling
        dist0 = np.linalg.norm(mkpts0, axis=1)  # Euclidean distances in the first image
        dist1 = np.linalg.norm(mkpts1, axis=1)  # Euclidean distances in the second image
        scale_ratios = dist1 / dist0  # Ratios of distances to estimate scale

        # Prepare data for RANSAC regression
        X = mkpts0  # Independent variables: coordinates from the first image
        y = np.vstack([x_diff, y_diff, scale_ratios]).T  # Dependent variables: translations and scale

        # Initialize and fit RANSAC for robust parameter estimation
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(X, y)

        # Extract the inlier mask from RANSAC
        inlier_mask = ransac.inlier_mask_

        # Calculate mean translations and scale using inliers
        tx = np.mean(x_diff[inlier_mask])  # Mean translation along X-axis
        ty = np.mean(y_diff[inlier_mask])  # Mean translation along Y-axis
        scale = np.mean(scale_ratios[inlier_mask])  # Mean scale from inliers

        # Construct the homography matrix for scaling and translations
        H = np.array([
            [scale, 0, tx],  # Scaling and X translation
            [0, scale, ty],  # Scaling and Y translation
            [0, 0, 1]        # Homogeneous coordinates
        ])
        return H
    else:
        # Print an error message if there are insufficient matches
        print('Not enough matches were found.')
        return None


def calculate_H_with_loftr_perspective(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0, confidence=0.99):
    """
    Calculates a perspective homography matrix between two images using LoFTR and RANSAC.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.
        confidence (float): Confidence level for the estimated homography. Defaults to 0.99.

    Returns:
        H (numpy.ndarray): The 3x3 perspective homography matrix if successfully computed.
        status (numpy.ndarray): Array indicating inliers and outliers (1 for inliers, 0 for outliers).
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors, normalize to [0, 1], and move them to the device
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize the LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare the input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image (query)
        "image1": img2_gray,  # Second image (reference)
    }

    # Perform feature matching using LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence_scores = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on confidence threshold
    high_confidence = confidence_scores > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Display the matched image for visualization
    matched_img = draw_matches(img1, img2, mkpts0, mkpts1)  # Function to visualize matches
    matched_img_show = cv2.resize(matched_img, (1920, 1080))  # Resize for better display
    cv2.imshow("Stitched Image", matched_img_show)
    
    # Wait for the "space" key to proceed
    while True:
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press
        if key == 32:  # ASCII code for "space"
            break

    # Estimate the perspective homography matrix using RANSAC
    if len(mkpts0) >= 4:  # At least 4 matches are required for homography estimation
        H, status = cv2.findHomography(
            mkpts0, mkpts1, cv2.RANSAC, ransac_residual_threshold, confidence=confidence
        )
        
        if H is not None:
            return H, status  # Return the homography matrix and inlier status
        else:
            print("Homography could not be estimated.")  # Handle cases where homography fails
            return None, None
    else:
        print("Not enough matches were found.")  # Handle cases with insufficient matches
        return None, None


def calculate_H_with_loftr_xy_scale_rotation_translation(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    """
    Calculates a homography matrix that includes translation, scaling, and rotation
    between two images using LoFTR and RANSAC.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.

    Returns:
        H (numpy.ndarray): A 3x3 homography matrix representing translation, scaling, and rotation.
        None: If not enough matches are found.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors, normalize to [0, 1], and move them to the device
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize the LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare the input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image (query)
        "image1": img2_gray,  # Second image (reference)
    }

    # Perform feature matching using LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on confidence threshold
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Ensure there are enough matches for RANSAC to work
    if len(mkpts0) > 4:
        # Calculate differences in X and Y coordinates for translation estimation
        x_diff = mkpts1[:, 0] - mkpts0[:, 0]
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]

        # Use RANSAC to identify inliers for robust transformation estimation
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(mkpts0, np.vstack([x_diff, y_diff]).T)  # Fit RANSAC using matched points
        inlier_mask = ransac.inlier_mask_

        # Filter inlier matches
        mkpts0_inliers = mkpts0[inlier_mask]
        mkpts1_inliers = mkpts1[inlier_mask]

        # Calculate centroids of the inlier points
        centroid0 = np.mean(mkpts0_inliers, axis=0)  # Centroid of points in the first image
        centroid1 = np.mean(mkpts1_inliers, axis=0)  # Centroid of points in the second image

        # Compute scaling using the average distance to centroids
        dist0 = np.linalg.norm(mkpts0_inliers - centroid0, axis=1)
        dist1 = np.linalg.norm(mkpts1_inliers - centroid1, axis=1)
        scale = np.mean(dist1) / np.mean(dist0)  # Average scale factor

        # Calculate mean translations for X and Y directions
        tx = centroid1[0] - centroid0[0]  # X translation
        ty = centroid1[1] - centroid0[1]  # Y translation

        # Estimate rotation using the angle difference between matched inlier points
        angles0 = np.arctan2(mkpts0_inliers[:, 1] - centroid0[1], mkpts0_inliers[:, 0] - centroid0[0])
        angles1 = np.arctan2(mkpts1_inliers[:, 1] - centroid1[1], mkpts1_inliers[:, 0] - centroid1[0])
        angle_diffs = angles1 - angles0
        rotation_angle = np.mean(angle_diffs)  # Mean rotation angle in radians

        # Construct the homography matrix
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        H = np.array([
            [scale * cos_theta, -scale * sin_theta, tx],  # Scaling and rotation with X translation
            [scale * sin_theta,  scale * cos_theta, ty],  # Scaling and rotation with Y translation
            [0, 0, 1]                                    # Homogeneous coordinates
        ])
        return H
    else:
        print('Not enough matches were found.')  # Handle cases with insufficient matches
        return None


def calculate_H_with_limited_perspective(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    """
    Calculates a homography matrix that includes translation and limited perspective adjustments 
    between two images using LoFTR and RANSAC.

    Args:
        img1 (numpy.ndarray): The first image (query image).
        img2 (numpy.ndarray): The second image (reference image).
        ratio_thresh (float): Confidence ratio threshold for filtering matches. Defaults to 0.75.
        ransac_residual_threshold (float): Residual threshold for RANSAC. Defaults to 5.0.

    Returns:
        H (numpy.ndarray): A 3x3 homography matrix representing translation and limited perspective adjustments.
        None: If not enough matches are found.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors, normalize to [0, 1], and move them to the device
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize the LoFTR matcher with pretrained outdoor weights
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires single-channel input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    # Prepare the input dictionary for LoFTR
    input_dict = {
        "image0": img1_gray,  # First image (query)
        "image1": img2_gray,  # Second image (reference)
    }

    # Perform feature matching using LoFTR
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints and their confidence scores
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Keypoints in the first image
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Keypoints in the second image
    confidence = correspondences['confidence'].cpu().numpy()  # Confidence scores for matches

    # Filter matches based on confidence threshold
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]  # Filtered keypoints in the first image
    mkpts1 = mkpts1[high_confidence]  # Filtered keypoints in the second image

    # Ensure there are enough matches for RANSAC to work
    if len(mkpts0) > 4:
        # Calculate differences in X and Y coordinates for translation estimation
        x_diff = mkpts1[:, 0] - mkpts0[:, 0]
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]

        # Reshape data for RANSAC regression
        X = mkpts0  # Independent variables: coordinates from the first image
        y = np.vstack([x_diff, y_diff]).T  # Dependent variables: translations

        # Initialize and fit RANSAC for robust translation estimation
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(X, y)

        # Extract the inlier mask from RANSAC
        inlier_mask = ransac.inlier_mask_

        # Calculate mean translations along X and Y using inliers
        tx = np.mean(x_diff[inlier_mask])  # Mean translation along X-axis
        ty = np.mean(y_diff[inlier_mask])  # Mean translation along Y-axis

        # Estimate small perspective offsets (limited perspective distortion)
        # Use inliers to compute perspective terms
        p1 = np.mean((mkpts1[inlier_mask, 0] - tx) / mkpts0[inlier_mask, 0])  # Perspective in X
        p2 = np.mean((mkpts1[inlier_mask, 1] - ty) / mkpts0[inlier_mask, 1])  # Perspective in Y

        # Construct the homography matrix with limited perspective adjustments
        H = np.array([
            [1, 0, tx],   # Translation and identity for X-axis
            [0, 1, ty],   # Translation and identity for Y-axis
            [p1, p2, 1]   # Perspective terms
        ])
        return H
    else:
        # Print a message if not enough matches are found
        print('Not enough matches were found.')
        return None



# ----------stitch image pair (two images)----------
def stitch_pair(image1, image2, scale=0.3, roi_width_start=0.1, roi_width_end=0.9):
    """
    Stitches two images together using feature matching and homography estimation.

    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.
        roi_width_start (float): Start of the ROI (Region of Interest) width for feature extraction. Defaults to 0.1.
        roi_width_end (float): End of the ROI width for feature extraction. Defaults to 0.9.

    Returns:
        numpy.ndarray: The stitched image.
    """
    # Resize images for faster processing
    image1_scaled = cv2.resize(image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale)))
    image2_scaled = cv2.resize(image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale)))

    # Detect features within the ROI and compute descriptors
    # kp1, desc1 = detect_and_describe(image1_scaled)
    # kp2, desc2 = detect_and_describe(image2_scaled)
    kp1, desc1 = detect_and_describe(
        image1_scaled[:, int(image1_scaled.shape[1] * roi_width_start):int(image1_scaled.shape[1] * roi_width_end)],
        method='sift'
    )
    kp2, desc2 = detect_and_describe(
        image2_scaled[:, int(image2_scaled.shape[1] * roi_width_start):int(image2_scaled.shape[1] * roi_width_end)],
        method='sift'
    )

    # Match features and compute homography
    kp1, kp2, matches = match_features(kp1, desc1, kp2, desc2, lowe_ratio=0.7, cross_check=True)
    # H, status = find_homography(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
    # H, status = find_similarity_transformation(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
    H, status = find_similarity_transformation_noscale(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
    # H, status = find_translation_transformation(kp1, kp2, matches, primary_axis='y')

    # Visualize matches (optional)
    matched_result = cv2.drawMatches(image1_scaled, kp1, image2_scaled, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # display_matches(matched_result, 1)

    if H is None:
        print("Homography not found.")
        return None

    # Adjust homography to account for scaling
    H[0, 2] *= (1 / scale)
    H[1, 2] *= (1 / scale)
    H[2, 0] *= scale
    H[2, 1] *= scale

    # Calculate the size of the stitched output image
    points = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)
    all_points = np.concatenate((points, transformed_points), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # Compute translation to bring the image into the positive coordinate space
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp the first image
    output_img = cv2.warpPerspective(image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

    # Paste the second image into the stitched output
    start_x = translation_dist[0]
    start_y = translation_dist[1]
    end_x = start_x + image2.shape[1]
    end_y = start_y + image2.shape[0]

    end_x = min(end_x, output_img.shape[1])  # Ensure bounds do not exceed output dimensions
    end_y = min(end_y, output_img.shape[0])

    output_img[start_y:end_y, start_x:end_x] = image2[:(end_y-start_y), :(end_x-start_x)]

    # Create a mask and find contours to crop the output
    grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour and its bounding box
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop the image to the bounding box
        cropped_img = output_img[y:y + h, x:x + w]

        # Display the final stitched image (optional)
        # display_matches(cropped_img, 1)
        
        return cropped_img
    else:
        print("Could not find any contours to crop the stitched image.")
        return output_img


def stitch_pair_loftr(image1, image2, scale=0.3, roi_width_start=0.1, roi_width_end=0.9):
    """
    Stitches two images together using LoFTR-based feature matching and perspective homography.

    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.
        roi_width_start (float): Start of the ROI (Region of Interest) width for feature extraction. Defaults to 0.1.
        roi_width_end (float): End of the ROI width for feature extraction. Defaults to 0.9.

    Returns:
        numpy.ndarray: The stitched image.
        None: If homography cannot be estimated.
    """
    # Resize images for faster processing
    image1_scaled = cv2.resize(image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale)))
    image2_scaled = cv2.resize(image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale)))

    # Detect features and compute descriptors using LoFTR
    # Uncomment the following line to use a different homography calculation method:
    # H = calculate_H_with_limited_perspective(image1_scaled, image2_scaled)
    H, status = calculate_H_with_loftr_perspective(image1_scaled, image2_scaled)

    if H is None:
        print("Homography not found.")
        return None

    # Adjust homography for scaling
    H[0, 2] *= (1 / scale)  # Adjust X translation for scaling
    H[1, 2] *= (1 / scale)  # Adjust Y translation for scaling
    H[2, 0] *= scale        # Adjust X perspective terms for scaling
    H[2, 1] *= scale        # Adjust Y perspective terms for scaling

    # Calculate the size of the stitched output image
    points = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)
    all_points = np.concatenate((points, transformed_points), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # Compute translation to bring the image into the positive coordinate space
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp the first image into the stitched canvas
    output_img = cv2.warpPerspective(image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

    # Paste the second image into the stitched canvas
    start_x = translation_dist[0]
    start_y = translation_dist[1]
    end_x = start_x + image2.shape[1]
    end_y = start_y + image2.shape[0]

    # Ensure the region to paste into matches the output dimensions
    end_x = min(end_x, output_img.shape[1])
    end_y = min(end_y, output_img.shape[0])

    output_img[start_y:end_y, start_x:end_x] = image2[:(end_y-start_y), :(end_x-start_x)]

    # Create a mask to identify non-black regions in the stitched image
    grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

    # Find contours to crop the stitched output
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour and its bounding box
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        margin = 0

        # Calculate the new crop boundaries
        x_new = max(x + margin, 0)
        y_new = max(y + margin, 0)
        w_new = min(w - 2 * margin, output_img.shape[1] - x_new)
        h_new = min(h - 2 * margin, output_img.shape[0] - y_new)

        # Crop the image to the bounding box
        cropped_img = output_img[y_new:y_new + h_new, x_new:x_new + w_new]

        # Display the stitched result (optional)
        display_matches(cropped_img, 1)

        return cropped_img
    else:
        print("Could not find any contours to crop the stitched image.")
        return output_img


def stitch_pair_loftr_v2(image1, image2, scale=0.3, roi_width_start=0.1, roi_width_end=0.9):
    """
    Stitches two images using LoFTR-based feature matching and homography estimation.

    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.
        roi_width_start (float): Start of the ROI (Region of Interest) width for feature extraction. Defaults to 0.1.
        roi_width_end (float): End of the ROI width for feature extraction. Defaults to 0.9.

    Returns:
        numpy.ndarray: The stitched image.
        None: If homography cannot be estimated.
    """
    # Resize images for faster processing
    image1_scaled = cv2.resize(image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale)))
    image2_scaled = cv2.resize(image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale)))

    # Compute the homography matrix using LoFTR feature matching
    # Uncomment the following line for alternative homography computation:
    # H = calculate_H_with_loftr_xy_translation(image1_scaled, image2_scaled)
    H, status = calculate_H_with_loftr_perspective(image1_scaled, image2_scaled)

    if H is None:
        print("Homography not found.")
        return None

    # Adjust homography to account for the scaling applied to the images
    H[0, 2] *= (1 / scale)  # Adjust translation in X
    H[1, 2] *= (1 / scale)  # Adjust translation in Y

    # Calculate the size of the output image canvas
    points = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)
    all_points = np.concatenate((points, transformed_points), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    # Translate the homography to ensure all coordinates are positive
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp the first image using the translated homography
    output_img = cv2.warpPerspective(image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

    # Paste the second image into the output canvas
    start_x = translation_dist[0]
    start_y = translation_dist[1]
    end_x = min(start_x + image2.shape[1], output_img.shape[1])
    end_y = min(start_y + image2.shape[0], output_img.shape[0])
    output_img[start_y:end_y, start_x:end_x] = image2[:end_y - start_y, :end_x - start_x]

    # Create a binary mask to identify non-black pixels
    grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

    # Find contours to determine the stitched area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop the stitched image to the bounding box of the largest contour
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_img = output_img[y:y + h, x:x + w]
        return cropped_img
    else:
        print("Could not find any contours to crop the stitched image.")
        return output_img



# ----------stitch group of images----------
def hierarchical_stitch(images, scale=0.3):
    """
    Stitches a list of images hierarchically. At each iteration, pairs of images are stitched
    together until a single final image remains.

    Args:
        images (list): List of images to be stitched.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.

    Returns:
        numpy.ndarray: The final stitched image.
    """
    while len(images) > 1:
        temp_images = []

        # Iterate through pairs of images
        for i in range(0, len(images), 2):
            print(f'Stitching images {i} and {i + 1}...')
            image1 = images[i]
            image2 = images[i + 1] if i + 1 < len(images) else image1  # Use the same image if no pair exists
            stitched = stitch_pair(image1, image2, scale)  # Stitch the current pair of images
            if stitched is not None:
                temp_images.append(stitched)
            else:
                temp_images.append(image1)  # If stitching failed, use the first image

        images = temp_images  # Prepare for the next round of stitching
        print(f'Number of images left: {len(images)}')

    return images[0]  # The final stitched image


def stitch_images(images, scale=0.3, eps=20, min_samples=3):
    """
    Stitches a list of images sequentially by aligning each subsequent image to the base image.

    Args:
        images (list): List of images to be stitched.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.
        eps (float): Distance parameter for DBSCAN clustering (not used in this implementation).
        min_samples (int): Minimum samples for DBSCAN clustering (not used in this implementation).

    Returns:
        numpy.ndarray: The final stitched image.
        None: If stitching is not possible.
    """
    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return None

    base_img = images[0]  # Start with the first image as the base
    for i in range(1, len(images)):
        print(f'Stitching image {i}...')

        # Resize the base image and the next image for faster processing
        base_img_scaled = cv2.resize(base_img, (int(base_img.shape[1] * scale), int(base_img.shape[0] * scale)))
        next_img = images[i]
        next_img_scaled = cv2.resize(next_img, (int(next_img.shape[1] * scale), int(next_img.shape[0] * scale)))

        # Detect features and compute descriptors
        kp1, desc1 = detect_and_describe(base_img_scaled)
        kp2, desc2 = detect_and_describe(next_img_scaled)

        # Match features
        kp1, kp2, matches = match_features(kp1, desc1, kp2, desc2, lowe_ratio=0.7, cross_check=True)

        # Visualize matches (optional)
        matched_result = cv2.drawMatches(base_img_scaled, kp1, next_img_scaled, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Uncomment one of the following lines to compute the homography:
        # H, status = find_homography(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
        # H, status = find_similarity_transformation(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
        # H, status = find_similarity_transformation_noscale(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
        H, status = find_translation_transformation(kp1, kp2, matches, primary_axis='y')

        if H is None:
            print("Homography not found.")
            continue

        # Adjust homography for scaling
        H[0, 2] *= (1 / scale)  # Adjust X translation
        H[1, 2] *= (1 / scale)  # Adjust Y translation

        # Calculate the size of the new stitched canvas
        points = np.float32([[0, 0], [0, base_img.shape[0]], [base_img.shape[1], base_img.shape[0]], [base_img.shape[1], 0]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, H)
        all_points = np.concatenate((points, transformed_points), axis=0)
        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        # Translate the homography to bring all coordinates into the positive space
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp the base image onto the new canvas
        output_img = cv2.warpPerspective(base_img, H_translation.dot(H), (x_max - x_min, y_max - y_min))

        # Paste the next image onto the canvas
        output_img[translation_dist[1]:translation_dist[1] + next_img.shape[0], translation_dist[0]:translation_dist[0] + next_img.shape[1]] = next_img

        # Create a binary mask to identify non-black regions
        grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

        # Find contours to crop the stitched output
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            base_img = output_img[y:y + h, x:x + w]  # Crop to the bounding box
        else:
            print("Could not find any contours to crop the stitched image.")
            base_img = output_img

    return base_img


def stitch_images_loftr(images, scale=0.3, eps=20, min_samples=3):
    """
    Stitches a list of images sequentially using LoFTR for feature matching and perspective transformation.

    Args:
        images (list): List of images to stitch.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.
        eps (float): Unused parameter for DBSCAN clustering (can be removed or customized later).
        min_samples (int): Unused parameter for DBSCAN clustering (can be removed or customized later).

    Returns:
        numpy.ndarray: The final stitched image.
        None: If stitching is not possible.
    """
    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return None

    # Start with the first image as the base image
    base_img = images[0]
    
    for i in range(1, len(images)):
        print(f'Stitching image {i}...')

        # Resize the base image and the next image for faster processing
        base_img_scaled = cv2.resize(base_img, (int(base_img.shape[1] * scale), int(base_img.shape[0] * scale)))
        next_img = images[i]
        next_img_scaled = cv2.resize(next_img, (int(next_img.shape[1] * scale), int(next_img.shape[0] * scale)))

        # Compute the homography matrix using LoFTR
        # Uncomment the following line for limited translation transformation:
        # H = calculate_H_with_loftr_y_translation(base_img_scaled, next_img_scaled)
        H, status = calculate_H_with_loftr_perspective(base_img_scaled, next_img_scaled)

        if H is None:
            print("Homography not found.")
            continue

        # Adjust homography for scaling
        H[0, 2] *= (1 / scale)  # Adjust X translation
        H[1, 2] *= (1 / scale)  # Adjust Y translation
        H[2, 0] *= scale        # Adjust perspective X terms
        H[2, 1] *= scale        # Adjust perspective Y terms

        # Calculate the size of the new canvas to fit both images
        points = np.float32([[0, 0], [0, base_img.shape[0]], [base_img.shape[1], base_img.shape[0]], [base_img.shape[1], 0]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, H)
        all_points = np.concatenate((points, transformed_points), axis=0)
        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        # Translate the homography to shift all coordinates into positive space
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp the base image into the new canvas
        output_img = cv2.warpPerspective(base_img, H_translation.dot(H), (x_max - x_min, y_max - y_min))

        # Overlay the next image into the canvas
        output_img[translation_dist[1]:translation_dist[1] + next_img.shape[0], translation_dist[0]:translation_dist[0] + next_img.shape[1]] = next_img

        # Create a binary mask to identify the stitched area
        grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

        # Find contours to crop the stitched output
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour and crop the image to its bounding box
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cropped_img = output_img[y:y + h, x:x + w]
            base_img = cropped_img  # Update the base image for the next iteration
        else:
            print("Could not find any contours to crop the stitched image.")
            base_img = output_img  # Use the current output as is if cropping fails

    return base_img


def hierarchical_stitch_loftr(images, scale=0.3):
    """
    Stitches a list of images hierarchically using LoFTR for feature matching and homography estimation.
    At each iteration, pairs of images are stitched together until a single image remains.

    Args:
        images (list): List of input images to stitch.
        scale (float): Scale factor for resizing images to speed up processing. Defaults to 0.3.

    Returns:
        numpy.ndarray: The final stitched image.
    """
    while len(images) > 1:
        temp_images = []  # Temporary list to store stitched images in the current iteration

        # Process images in pairs
        for i in range(0, len(images), 2):
            print(f'Stitching images {i} and {i + 1}...')

            # Select the current pair of images
            image1 = images[i]
            image2 = images[i + 1] if i + 1 < len(images) else image1  # If no pair, reuse the same image

            # Stitch the pair using the LoFTR-based stitching function
            stitched = stitch_pair_loftr(image1, image2, scale)
            if stitched is not None:
                temp_images.append(stitched)

                # Display the stitched image and wait for user interaction
                stitched_show = cv2.resize(stitched, (480, 1080))  # Resize for display
                cv2.imshow("Stitched Image", stitched_show)
                while True:
                    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
                    if key == 32:  # ASCII code for "space"
                        break  # Exit the display loop when the spacebar is pressed
                cv2.destroyAllWindows()  # Close the display window

            else:
                temp_images.append(image1)  # If stitching fails, keep the first image

        # Update the list of images for the next round of stitching
        images = temp_images
        print(f'Number of images left: {len(images)}')

    return images[0]  # Return the final stitched image


# ===========main function==========

# Define the folder path containing images to be stitched

# folder_path = r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton\plot\ponder_9A\test\one_image'
folder_path = '/blue/lift-phenomics/zhengkun.li/peanut_project/image_stitching/data/tifton/plot/ponder_9A/test/image'
# folder_path  = "/blue/lift-phenomics/zhengkun.li/peanut_project/image_stitching/data/tifton/plot/ponder_9A/test"
# folder_path = r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton\plot\ponder_9A\20231024_Ponder_9A_4_9_left'

# Load images from the specified folder
images = load_images_from_folder(folder_path)

# Check if a CUDA-enabled GPU is available; if not, fall back to the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure images were successfully loaded
if images:
    # Perform hierarchical stitching using LoFTR
    stitched_image = hierarchical_stitch_loftr(images, scale=0.3)
    # stitched_image = stitch_images_loftr(images, scale=0.3, eps=20, min_samples=3)

    # Display and save the final stitched image
    if stitched_image is not None:
        # cv2.imshow('Stitched Image-loftr', stitched_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('stitched_image_loftr_hierarchical_stitch-test.jpg', stitched_image)
        cv2.imwrite('stitched_image_loftr_xy_perspective.jpg', stitched_image)
        
else:
    # Print an error message if no images were found in the folder
    print("No images to stitch.")
