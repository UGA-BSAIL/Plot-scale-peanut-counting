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
import sys


def load_images_from_folder(folder, width_roi_start=0.2, width_roi_end=0.8):
    files = os.listdir(folder)
    # ä½¿ç¨æ­£åè¡¨è¾¾å¼æåæä»¶åä¸­çæ°å­ï¼è¿è¡æåº
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()), reverse=True)  # éåºæåº

    images = []
    for filename in files:
        # print(filename)
        img = cv2.imread(os.path.join(folder, filename))
        # img = cv2.resize(img, None, fx=image_scale, fy=image_scale)
        if img is not None:
            height, width, _ = img.shape
            crop_start = int(width * width_roi_start)
            crop_end = int(width * width_roi_end)
            cropped_img = img[:, crop_start:crop_end]
            images.append(cropped_img)
    return images

def detect_and_describe(image, method='sift'):
    if method == 'sift':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
    elif method == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image, None)

    elif method == 'orb':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

    elif method == 'brisk':
        brisk = cv2.BRISK_create()
        keypoints, descriptors = brisk.detectAndCompute(image, None)

    elif method == 'akaze':
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(image, None)

    elif method == 'kaze':
        kaze = cv2.KAZE_create()
        keypoints, descriptors = kaze.detectAndCompute(image, None)

    elif method == 'daisy':
        daisy = cv2.xfeatures2d.DAISY_create()
        keypoints = daisy.detect(image)
        keypoints, descriptors = daisy.compute(image, keypoints)

    elif method == 'lucid':
        lucid = cv2.xfeatures2d.LUCID_create()
        keypoints = lucid.detect(image)
        keypoints, descriptors = lucid.compute(image, keypoints)
        
    elif method == 'latch':
        latch = cv2.xfeatures2d.LATCH_create()
        keypoints = latch.detect(image)
        keypoints, descriptors = latch.compute(image, keypoints)

    elif method == 'freak':
        freak = cv2.xfeatures2d.FREAK_create()
        keypoints = freak.detect(image)
        keypoints, descriptors = freak.compute(image, keypoints)

    elif method == 'superpoint':
        superpoint = cv2.superres.DenseOpticalFlow_ext_createSuperPoint()
        keypoints = superpoint.detect(image)
        keypoints, descriptors = superpoint.compute(image, keypoints)

    elif method == 'd2net':
        d2net = cv2.dnn.readNetFromTorch('d2net/d2_tf.pth')
        keypoints = d2net.detect(image)
        keypoints, descriptors = d2net.compute(image, keypoints)

    elif method == 'sift_gpu': # SIFT on GPU
        sift = cv2.cuda.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = sift.detect(gray_gpu, None)
        keypoints, descriptors = sift.compute(gray_gpu, keypoints)

    elif method == 'surf_gpu': # SURF on GPU
        surf = cv2.cuda.SURF_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = surf.detect(gray_gpu, None)
        keypoints, descriptors = surf.compute(gray_gpu, keypoints)

    elif method == 'orb_gpu': # ORB on GPU
        orb = cv2.cuda_ORB.create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = orb.detect(gray_gpu, None)
        keypoints, descriptors = orb.compute(gray_gpu, keypoints)

    elif method == 'fast_gpu': # FAST on GPU
        fast = cv2.cuda.FastFeatureDetector_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = fast.detect(gray_gpu, None)
        keypoints, descriptors = fast.compute(gray_gpu, keypoints)

    elif method == 'brisk_gpu': # BRISK on GPU
        brisk = cv2.cuda.createBRISK()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = brisk.detect(gray_gpu, None)
        keypoints, descriptors = brisk.compute(gray_gpu, keypoints)

    elif method == 'akaze_gpu': # AKAZE on GPU
        akaze = cv2.cuda.createAKAZE()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = akaze.detect(gray_gpu, None)
        keypoints, descriptors = akaze.compute(gray_gpu, keypoints)

    elif method == 'kaze_gpu': # KAZE on GPU
        kaze = cv2.cuda.createKAZE()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = kaze.detect(gray_gpu, None)
        keypoints, descriptors = kaze.compute(gray_gpu, keypoints)

    elif method == 'superpoint_gpu': # SuperPoint on GPU
        superpoint = cv2.cuda_SuperPoint.create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = superpoint.detect(gray_gpu, None)
        keypoints, descriptors = superpoint.compute(gray_gpu, keypoints)

    elif method == 'd2net_gpu': # D2-Net on GPU
        d2net = cv2.cuda_D2Net.create('d2net/d2_tf.pth')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        keypoints = d2net.detect(gray_gpu, None)
        keypoints, descriptors = d2net.compute(gray_gpu, keypoints)

    else:
        raise ValueError(f"Unknown feature detection method: {method}")
    
    return keypoints, descriptors

def match_features(kp1, desc1, kp2, desc2, lowe_ratio=0.75, cross_check=True):
    # FLANN parameters and matcher initialization
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Find initial matches using k-nearest neighbors (with k=2)
    # print(type(desc1), type(desc2))
    desc1 = np.float32(desc1)
    desc2 = np.float32(desc2)

    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)
    
    # If cross-check is desired
    if cross_check:
        good_matches_cc = []
        matches_reverse = matcher.knnMatch(desc2, desc1, k=1)
        for m in good_matches:
            if matches_reverse[m.trainIdx][0].trainIdx == m.queryIdx:
                good_matches_cc.append(m)
        good_matches = good_matches_cc

    return kp1, kp2, good_matches

def find_homography(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.99):
    if len(matches) >= 4:  # We need at least 4 matches to compute homography
        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find homography using RANSAC
        H, status = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold=max_reproj_error, confidence=confidence)
        return H, status
    else:
        return None, None 

def find_similarity_transformation(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.99):
    if len(matches) >= 4:  # We need at least 4 matches to compute a transformation
        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Estimate the similarity transformation matrix using RANSAC
        M, status = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC, 
                                                 ransacReprojThreshold=max_reproj_error, 
                                                 confidence=confidence)
        
        # Convert the matrix to a 3x3 matrix to make it compatible with the homography functions
        if M is not None:
            H = np.vstack([M, [0,0,1]])  # Append a row [0,0,1] to convert to homography format
            return H, status
        else:
            return None, None
    else:
        return None, None
    
def find_similarity_transformation_noscale(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.99):
    if len(matches) >= 4:  # We need at least 4 matches to compute a transformation
        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Estimate the similarity transformation matrix using RANSAC
        M, status = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC, 
                                                 ransacReprojThreshold=max_reproj_error, 
                                                 confidence=confidence)
        
        # Normalize the rotation matrix part to exclude any scaling
        if M is not None:
            T = M[:,2]  # Translation component
            R = M[:,0:2]  # Rotation and scaling component
            # Normalize to exclude scaling from rotation matrix
            norm = np.sqrt(np.linalg.det(R))
            R /= norm
            # Recombine to form the affine matrix without scaling
            M = np.hstack([R, T.reshape(-1, 1)])
            H = np.vstack([M, [0,0,1]])  # Append a row [0,0,1] to convert to homography format
            return H, status
        else:
            return None, None
    else:
        return None, None
    
def find_translation_transformation(kp1, kp2, matches, primary_axis='y'):
    if len(matches) >= 2:  # At least 2 matches are needed to compute a translation
        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Compute the difference in the primary axis (x or y)
        if primary_axis == 'x':
            translations = points2[:, 0] - points1[:, 0]
        else:
            translations = points2[:, 1] - points1[:, 1]

        # Estimate the main translation using the median to minimize the effect of outliers
        main_translation = np.median(translations)

        # Create the translation matrix
        if primary_axis == 'x':
            T = np.array([[1, 0, main_translation],
                          [0, 1, 0],
                          [0, 0, 1]])
        else:
            T = np.array([[1, 0, 0],
                          [0, 1, main_translation],
                          [0, 0, 1]])

        return T, None  # Status is not applicable here
    else:
        return None, None

def display_matches(matched_result, image_index, max_width=1280, max_height=2400):
    # Calculate the aspect ratio of the image
    h, w = matched_result.shape[:2]
    aspect_ratio = w / h

    # Determine the new width and height based on the maximum size
    if h > max_height or w > max_width:
        # Calculate the new size based on the max height while maintaining the aspect ratio
        if max_height / h < max_width / w:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
    else:
        new_width, new_height = w, h  # Use the original size if it's within the limits

    # Resize the image
    resized_image = cv2.resize(matched_result, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Show the image
    # cv2.imshow(f'Matched features between image {image_index - 1} and image {image_index}', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def stitch_pair(image1, image2, scale=0.3, roi_width_start=0.1, roi_width_end=0.9):
    # Resize images for faster processing
    image1_scaled = cv2.resize(image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale)))
    image2_scaled = cv2.resize(image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale)))

    # Detect features and compute descriptors
    # kp1, desc1 = detect_and_describe(image1_scaled)
    # kp2, desc2 = detect_and_describe(image2_scaled)
    kp1, desc1 = detect_and_describe(image1_scaled[:, int(image1_scaled.shape[1] * roi_width_start):int(image1_scaled.shape[1] * roi_width_end)], method='sift')
    kp2, desc2 = detect_and_describe(image2_scaled[:, int(image2_scaled.shape[1] * roi_width_start):int(image2_scaled.shape[1] * roi_width_end)], method='sift')

    # Match features and compute homography
    kp1, kp2, matches = match_features(kp1, desc1, kp2, desc2, lowe_ratio=0.7, cross_check=True)
    # H, status = find_homography(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
    # H, status = find_similarity_transformation(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
    H, status = find_similarity_transformation_noscale(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
    # H, status = find_translation_transformation(kp1, kp2, matches, primary_axis='y')

    # draw the matches
    matched_result = cv2.drawMatches(image1_scaled, kp1, image2_scaled, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # display_matches(matched_result, 1)

    if H is None:
        print("Homography not found.")
        return None

    # Adjust homography for scaling
    H[0, 2] *= (1 / scale)
    H[1, 2] *= (1 / scale)
    H[2, 0] *= scale
    H[2, 1] *= scale

    # Calculate the size of the new image
    points = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)
    all_points = np.concatenate((points, transformed_points), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

    # Ensure the region to paste into matches image2's dimensions
    start_x = translation_dist[0]
    start_y = translation_dist[1]
    end_x = start_x + image2.shape[1]
    end_y = start_y + image2.shape[0]

    # Check if the pasting coordinates exceed the output_img dimensions
    end_x = min(end_x, output_img.shape[1])
    end_y = min(end_y, output_img.shape[0])

    # Paste image2 into output_img
    output_img[start_y:end_y, start_x:end_x] = image2[:(end_y-start_y), :(end_x-start_x)]

    # Create a mask where non-black pixels are marked
    grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will be the contour of the stitched image
        max_contour = max(contours, key=cv2.contourArea)

        # Compute the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)

        margin = 0

        # Calculate new boundaries ensuring they are within the image dimensions
        x_new = max(x + margin, 0)
        y_new = max(y + margin, 0)
        w_new = min(w - 2 * margin, output_img.shape[1] - x_new)
        h_new = min(h - 2 * margin, output_img.shape[0] - y_new)

        # Crop the image to the new bounding box size, removing the margin
        cropped_img = output_img[y_new:y_new + h_new, x_new:x_new + w_new]

        display_matches(cropped_img, 1)
        
        return cropped_img
    else:
        print("Could not find any contours to crop the stitched image.")
        return output_img

def calculate_H_with_loftr_y_translation(img1, img2, ratio_thresh=0.75, ransac_residual_threshold=5.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert images to tensors and move to GPU
    img1_tensor = K.image_to_tensor(img1, keepdim=False).float().to(device) / 255.
    img2_tensor = K.image_to_tensor(img2, keepdim=False).float().to(device) / 255.

    # Initialize LoFTR matcher
    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    # Convert images to grayscale as LoFTR requires grayscale input
    img1_gray = K.color.rgb_to_grayscale(img1_tensor)
    img2_gray = K.color.rgb_to_grayscale(img2_tensor)

    input_dict = {
        "image0": img1_gray,  # query image
        "image1": img2_gray,  # reference image
    }

    # Perform feature matching
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched points
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    # Filter matches with high confidence
    high_confidence = confidence > ratio_thresh
    mkpts0 = mkpts0[high_confidence]
    mkpts1 = mkpts1[high_confidence]

    # Estimate translation along the Y-axis using RANSAC
    if len(mkpts0) > 4:
        y_diff = mkpts1[:, 1] - mkpts0[:, 1]  # Differences in Y coordinates

        # Reshape data for RANSAC
        X = mkpts0[:, 1].reshape(-1, 1)  # Use Y-coordinates of mkpts0 as independent variable
        y = y_diff.reshape(-1, 1)        # Differences as dependent variable

        # Initialize and fit RANSAC
        ransac = RANSACRegressor(residual_threshold=ransac_residual_threshold)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_

        # Use inliers to calculate mean translation
        ty = np.mean(y_diff[inlier_mask])  # Mean of inliers

        # Construct homography matrix that translates only on Y-axis
        H = np.array([
            [1, 0, 0],
            [0, 1, ty],
            [0, 0, 1]
        ])
        return H
    else:
        print('Not enough matches were found.')
        return None


def stitch_pair_loftr(image1, image2, scale=0.3, roi_width_start=0.1, roi_width_end=0.9):
    # Resize images for faster processing
    image1_scaled = cv2.resize(image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale)))
    image2_scaled = cv2.resize(image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale)))

    # Detect features and compute descriptors

    H = calculate_H_with_loftr_y_translation(image1_scaled, image2_scaled)


    if H is None:
        print("Homography not found.")
        return None

    # Adjust homography for scaling
    H[0, 2] *= (1 / scale)
    H[1, 2] *= (1 / scale)
    H[2, 0] *= scale
    H[2, 1] *= scale

    # Calculate the size of the new image
    points = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, H)
    all_points = np.concatenate((points, transformed_points), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(image1, H_translation.dot(H), (x_max - x_min, y_max - y_min))

    # Ensure the region to paste into matches image2's dimensions
    start_x = translation_dist[0]
    start_y = translation_dist[1]
    end_x = start_x + image2.shape[1]
    end_y = start_y + image2.shape[0]

    # Check if the pasting coordinates exceed the output_img dimensions
    end_x = min(end_x, output_img.shape[1])
    end_y = min(end_y, output_img.shape[0])

    # Paste image2 into output_img
    output_img[start_y:end_y, start_x:end_x] = image2[:(end_y-start_y), :(end_x-start_x)]

    # Create a mask where non-black pixels are marked
    grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will be the contour of the stitched image
        max_contour = max(contours, key=cv2.contourArea)

        # Compute the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)

        margin = 0

        # Calculate new boundaries ensuring they are within the image dimensions
        x_new = max(x + margin, 0)
        y_new = max(y + margin, 0)
        w_new = min(w - 2 * margin, output_img.shape[1] - x_new)
        h_new = min(h - 2 * margin, output_img.shape[0] - y_new)

        # Crop the image to the new bounding box size, removing the margin
        cropped_img = output_img[y_new:y_new + h_new, x_new:x_new + w_new]

        display_matches(cropped_img, 1)
        
        return cropped_img
    else:
        print("Could not find any contours to crop the stitched image.")
        return output_img

def hierarchical_stitch(images, scale=0.3):
    while len(images) > 1:
        temp_images = []

        for i in range(0, len(images), 2):
            # print(f'Stitching images {i} and {i + 1}...')
            image1 = images[i]
            image2 = images[i + 1] if i + 1 < len(images) else image1  # Use the same image if no pair exists
            stitched = stitch_pair(image1, image2, scale)
            if stitched is not None:
                temp_images.append(stitched)
            else:
                temp_images.append(image1)  # If stitching failed, use the first image

        images = temp_images  # Prepare for the next round
        print(f'Number of images left: {len(images)}')

    return images[0]  # The final stitched image

def stitch_images(images, scale=0.3, eps=20, min_samples=3):
    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return None

    base_img = images[0]
    for i in range(1, len(images)):
        print(f'Stitching image {i}...')

        base_img_scaled = cv2.resize(base_img, (int(base_img.shape[1] * scale), int(base_img.shape[0] * scale)))

        next_img = images[i]
        next_img_scaled = cv2.resize(next_img, (int(next_img.shape[1] * scale), int(next_img.shape[0] * scale)))

        kp1, desc1 = detect_and_describe(base_img_scaled)
        kp2, desc2 = detect_and_describe(next_img_scaled)
        
        # matches = match_features(desc1, desc2)
        kp1, kp2, matches = match_features(kp1, desc1, kp2, desc2, lowe_ratio=0.7, cross_check=True)

        # draw the matches
        matched_result = cv2.drawMatches(base_img_scaled, kp1, next_img_scaled, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # display_matches(matched_result, i)

        # H, status = find_homography(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
        # H, status = find_similarity_transformation(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
        # H, status = find_similarity_transformation_noscale(kp1, kp2, matches, max_reproj_error=3.0, confidence=0.995)
        H, status = find_translation_transformation(kp1, kp2, matches, primary_axis='y')

        if H is None:
            print("Homography not found.")
            continue

        # Adjust homography for scaling
        H[0, 2] *= (1 / scale)
        H[1, 2] *= (1 / scale)
        H[2, 0] *= scale
        H[2, 1] *= scale

        # Calculate the size of the new image
        points = np.float32([[0, 0], [0, base_img.shape[0]], [base_img.shape[1], base_img.shape[0]], [base_img.shape[1], 0]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, H)
        all_points = np.concatenate((points, transformed_points), axis=0)
        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        output_img = cv2.warpPerspective(base_img, H_translation.dot(H), (x_max - x_min, y_max - y_min))
        output_img[translation_dist[1]:translation_dist[1] + next_img.shape[0], translation_dist[0]:translation_dist[0] + next_img.shape[1]] = next_img

        # Create a mask where non-black pixels are marked
        grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which will be the contour of the stitched image
            max_contour = max(contours, key=cv2.contourArea)

            # Compute the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(max_contour)

            # Crop the image to the bounding box size
            cropped_img = output_img[y:y + h, x:x + w]
            base_img = cropped_img
        else:
            print("Could not find any contours to crop the stitched image.")
            base_img = output_img

    return base_img

def stitch_images_loftr(images, scale=0.3, eps=20, min_samples=3):
    if len(images) < 2:
        print("Need at least 2 images to stitch.")
        return None

    base_img = images[0]
    for i in range(1, len(images)):
        # print(f'Stitching image {i}...')

        base_img_scaled = cv2.resize(base_img, (int(base_img.shape[1] * scale), int(base_img.shape[0] * scale)))

        next_img = images[i]
        next_img_scaled = cv2.resize(next_img, (int(next_img.shape[1] * scale), int(next_img.shape[0] * scale)))
        
        H = calculate_H_with_loftr_y_translation(base_img_scaled, next_img_scaled)

        if H is None:
            print("Homography not found.")
            continue

        # Adjust homography for scaling
        H[0, 2] *= (1 / scale)
        H[1, 2] *= (1 / scale)
        H[2, 0] *= scale
        H[2, 1] *= scale

        # Calculate the size of the new image
        points = np.float32([[0, 0], [0, base_img.shape[0]], [base_img.shape[1], base_img.shape[0]], [base_img.shape[1], 0]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, H)
        all_points = np.concatenate((points, transformed_points), axis=0)
        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        output_img = cv2.warpPerspective(base_img, H_translation.dot(H), (x_max - x_min, y_max - y_min))
        output_img[translation_dist[1]:translation_dist[1] + next_img.shape[0], translation_dist[0]:translation_dist[0] + next_img.shape[1]] = next_img

        # Create a mask where non-black pixels are marked
        grayscale = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which will be the contour of the stitched image
            max_contour = max(contours, key=cv2.contourArea)

            # Compute the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(max_contour)

            # Crop the image to the bounding box size
            cropped_img = output_img[y:y + h, x:x + w]
            base_img = cropped_img
        else:
            print("Could not find any contours to crop the stitched image.")
            base_img = output_img

    return base_img

def hierarchical_stitch_loftr(images, scale=0.3):
    while len(images) > 1:
        temp_images = []

        for i in range(0, len(images), 2):
            print(f'Stitching images {i} and {i + 1}...')
            image1 = images[i]
            image2 = images[i + 1] if i + 1 < len(images) else image1  # Use the same image if no pair exists
            stitched = stitch_pair_loftr(image1, image2, scale)
            if stitched is not None:
                temp_images.append(stitched)
            else:
                temp_images.append(image1)  # If stitching failed, use the first image

        images = temp_images  # Prepare for the next round
        print(f'Number of images left: {len(images)}')

    return images[0]  # The final stitched image



# folder_path = r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton\plot\ponder_9A\test\one_image'
folder_path = '/blue/cli2/zhengkun.li/peanut_project/Analysis_result/SFM_MetaShape/left'
save_path = '/blue/cli2/zhengkun.li/peanut_project/Analysis_result/LoFTR-stitching/stitching-result/loftr-y-translation/second/left'

# check if the folder path exists
if not os.path.exists(folder_path):
    print("The folder path does not exist.")
    sys.exit()

# check if the save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get all subfolder names in the folder path
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

for subfolder in subfolders:
    print(f'Processing subfolder: {subfolder}')
    
    images = load_images_from_folder(subfolder)
    if images:
        stitched_image = hierarchical_stitch_loftr(images, scale=0.3)
        # stitched_image = stitch_images_loftr(images, scale=0.3, eps=20, min_samples=3)
        if stitched_image is not None:
            # cv2.imshow('Stitched Image-loftr', stitched_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # save the stitched image into the save_path
            subfolder_name = Path(subfolder).name
            save_filename = os.path.join(save_path, f"{subfolder_name}_stitched.jpg")
            cv2.imwrite(save_filename, stitched_image)
            print(f"Saved stitched image to {save_filename}")
        else:
            print("Stitching failed or no images to stitch in", subfolder)
    else:
        print(f"No images found in {subfolder}")
