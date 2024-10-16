import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import kornia as K
import kornia.feature as KF
import torch
import os
import csv
import re

def load_images_from_folder(folder, width_roi_start=0.2, width_roi_end=0.8, image_scale=0.3):
    files = os.listdir(folder)
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()))  # 升序排序
    # files.sort(key=lambda f: int(re.search(r'\d+', f).group()), reverse=True)  # 降序排序

    images = []
    for filename in files:
        print(filename)
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, None, fx=image_scale, fy=image_scale)
        if img is not None:
            height, width, _ = img.shape
            crop_start = int(width * width_roi_start)
            crop_end = int(width * width_roi_end)
            cropped_img = img[:, crop_start:crop_end]
            images.append(cropped_img)
    return images

def detect_keypoints(img1, img2, method='sift'):
    if method == 'sift':
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2
    
    elif method == 'orb':
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2
    
    elif method == 'brisk':
        brisk = cv2.BRISK_create()
        kp1, des1 = brisk.detectAndCompute(img1, None)
        kp2, des2 = brisk.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2
    else:
        raise ValueError('Invalid method. Use "sift" or "orb".')

def match_keypoints(kp1, des1, kp2, des2, method='sift'):
    if method == 'sift':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches
    
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return matches
    else:
        raise ValueError('Invalid method. Use "sift" or "orb".')

def detect_and_match_keypoints(img1, img2, method='sift'):

    if method == 'sift' or method == 'orb' or method == 'brisk':
        kp1, des1, kp2, des2 = detect_keypoints(img1, img2, method)
        matches = match_keypoints(kp1, des1, kp2, des2, method)
        return kp1, des1, kp2, des2, matches
    
    elif method == 'loftr':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img1_tensor = K.image_to_tensor(img1).to(device).float() / 255.
        img2_tensor = K.image_to_tensor(img2).to(device).float() / 255.

        loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)
        with torch.no_grad():
            pred = loftr(img1_tensor, img2_tensor)
        matches = pred['matches0']

        kp1 = KF.get_sift_keypoints_from_matches(pred['keypoints0'], matches)
        kp2 = KF.get_sift_keypoints_from_matches(pred['keypoints1'], matches)
        des1 = KF.get_sift_descriptors_from_matches(pred['descriptors0'], matches)
        des2 = KF.get_sift_descriptors_from_matches(pred['descriptors1'], matches)
        return kp1, des1, kp2, des2, matches

def calculate_homography(kp1, kp2, matches, good_match = 0.75, max_reproj_error=3.0, confidence=0.99):
    good_matches = []
    for m,n in matches:
        if m.distance < good_match * n.distance:
            good_matches.append(m)

    # update the kp1 and kp2 with good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, max_reproj_error, confidence)
    
    if homography is None:
        raise ValueError('Homography could not be calculated.')
    return homography

def stitch_images(img1, img2, method='sift', imshow=False):
    
    if method == 'simple':
        h, w, _ = img1.shape
        overlap = int(w / 10)
        stitched = np.zeros((h, w + overlap, 3), dtype=np.uint8)
        stitched[:, :w] = img1
        stitched[:, w:] = img2

        if imshow:
            cv2.imshow('Stitched Image', stitched)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return stitched
    
    elif method == 'sift' or method == 'orb' or method == 'brisk':
        kp1, des1, kp2, des2, matches = detect_and_match_keypoints(img1, img2, method)
        homography = calculate_homography(kp1, kp2, matches)
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        stitched = cv2.warpPerspective(img1, homography, (w1 + w2, h1))
        stitched[0:h2, 0:w2] = img2

        if imshow:  
            cv2.imshow('Stitched Image', stitched)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return stitched
    
    elif method == 'loftr':
        kp1, des1, kp2, des2, matches = detect_and_match_keypoints(img1, img2, method)
        homography = calculate_homography(kp1, kp2, matches)
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        stitched = cv2.warpPerspective(img1, homography, (w1 + w2, h1))
        stitched[0:h2, 0:w2] = img2

        if imshow:
            cv2.imshow('Stitched Image', stitched)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return stitched
    else:
        raise ValueError('Invalid method. Use "simple", "sift", "orb", "brisk", or "loftr".')

def split_image(img, overlap_rate, method = "horizontal"):
    h, w, _ = img.shape

    if method == "horizontal":
        img1 = img[:int(h * (0.5 + 0.5 * overlap_rate)), :]
        img2 = img[int(h * (0.5 - 0.5 * overlap_rate)):, :]

    elif method == "vertical":
        img1 = img[:, :int(w * (0.5 + 0.5 * overlap_rate))]
        img2 = img[:, int(w * (0.5 - 0.5 * overlap_rate)):]

    elif "random":
        if np.random.rand() < 0.5:
            img1 = img[:int(h * (0.5 + 0.5 * overlap_rate)), :]
            img2 = img[int(h * (0.5 - 0.5 * overlap_rate)):, :]
        else:
            img1 = img[:, :int(w * (0.5 + 0.5 * overlap_rate))]
            img2 = img[:, int(w * (0.5 - 0.5 * overlap_rate)):]

    else:
        raise ValueError('Invalid method. Use "horizontal", "vertical", or "random".')
    
    return img1, img2
    
def calculate_metrics(original, stitched, split_method='horizontal', overlap_rate=0.2):
    # Function to calculate evaluation metrics for the overlapping region
    overlap_org = None
    overlap_stitched = None
    if split_method == 'horizontal':
        h, w, _ = original.shape
        overlap_org = original[h*(0.5 - 0.5 * overlap_rate): h*(0.5 + 0.5 * overlap_rate), :]
        overlap_stitched = stitched[h*(0.5 - 0.5 * overlap_rate): h*(0.5 + 0.5 * overlap_rate), :]

    elif split_method == 'vertical':
        h, w, _ = original.shape
        overlap_org = original[:, w*(0.5 - 0.5 * overlap_rate): w*(0.5 + 0.5 * overlap_rate)]
        overlap_stitched = stitched[:, w*(0.5 - 0.5 * overlap_rate): w*(0.5 + 0.5 * overlap_rate)]

    elif split_method == 'random':
        print('Random split not supported for calculating metrics.')

    else:
        raise ValueError('Invalid method. Use "horizontal", "vertical", or "random".')
    
    # Function to calculate evaluation metrics
    mse_val = mse(overlap_org, overlap_stitched)
    psnr_val = cv2.PSNR(overlap_org, overlap_stitched)
    ssim_val = ssim(overlap_org, overlap_stitched, multichannel=True)

    return mse_val, psnr_val, ssim_val


# Main function to process images in a folder
def process_images(folder_path):

    # Load images from the folder
    images = load_images_from_folder(folder_path, width_roi_start=0.2, width_roi_end=0.8, image_scale=0.3)
    overlap_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Create a CSV file to store the evaluation metrics
    with open('stitch_comparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Image Name']
        for rate in overlap_rates:
            header += [f'MSE_{rate}', f'PSNR_{rate}', f'SSIM_{rate}']
        writer.writerow(header)

        for img in images:
            h, w, _ = img.shape

            # Split the image
            for rate in overlap_rates:
                img1, img2 = split_image(img, rate, method="horizontal")

                # Stitch the images
                stitched_img = stitch_images(img1, img2, method='sift', imshow=False)

                # Calculate evaluation metrics
                metrics = calculate_metrics(img, stitched_img)
                row = [f'Overlap Rate: {rate}']
                row.extend(metrics)
                writer.writerow(row)

# Call the main function with the path to your image folder
process_images(r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton\plot\ponder_9A\test\one_image')

