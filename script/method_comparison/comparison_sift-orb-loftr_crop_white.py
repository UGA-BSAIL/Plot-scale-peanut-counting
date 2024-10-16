import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.draw import polygon
import csv
import glob
import pandas as pd
from pathlib import Path
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.viz import draw_LAF_matches


def load_images_from_nested_folders(root_folder):
    image_list = []
    # Walk through all directories and files in the root folder
    for subdir, dirs, files in os.walk(root_folder):
        for file in glob.glob(os.path.join(subdir, '*.JPG')):  # Assuming the images are in jpg format
            img = cv2.imread(file)
            if img is not None:
                image_list.append(img)
    
    return image_list

def load_images_from_folder(folder, width_roi_start=0.2, width_roi_end=0.8, image_scale=0.3):
    files = os.listdir(folder)
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()))  # 升序排序
    # files.sort(key=lambda f: int(re.search(r'\d+', f).group()), reverse=True)  # 降序排序

    images = []
    for filename in files:
        # print(filename)
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, None, fx=image_scale, fy=image_scale)
        if img is not None:
            height, width, _ = img.shape
            crop_start = int(width * width_roi_start)
            crop_end = int(width * width_roi_end)
            cropped_img = img[:, crop_start:crop_end]
            images.append(cropped_img)
    return images

def random_variety(image):

    # covert the color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 随机改变图像的亮度和对比度
    alpha = np.random.uniform(0.8, 1.2)  # 对比度控制
    beta = np.random.uniform(-20, 20)   # 亮度控制
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 随机模糊图像
    ksize = np.random.choice([3, 5])
    blurred_image = cv2.GaussianBlur(adjusted_image, (ksize, ksize), 0)

    # add random shadow
    shadow_intensity = np.random.uniform(0.5, 0.8)
    shadowed_image = add_shadow_blocks(blurred_image, shadow_intensity=shadow_intensity)

    return shadowed_image


def random_homography(image):

    height, width = image.shape[:2]

    # 生成随机变换参数
    translation_x = np.random.uniform(-0.2 * width, 0.2 * width)
    translation_y = np.random.uniform(-0.2 * height, 0.2 * height)
    rotation_angle = np.random.uniform(-20, 20)
    scale = np.random.uniform(0.9, 1.4)
    shear_x = np.random.uniform(-0.2, 0.2)
    shear_y = np.random.uniform(-0.2, 0.2)
    perspective = np.random.uniform(0, 0.001)

    # 构建变换矩阵
    T = np.float32([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
    center = (width / 2, height / 2)
    R = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    R = np.vstack([R, [0, 0, 1]])
    S = np.float32([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]])
    P = np.float32([[1, 0, 0], [0, 1, 0], [perspective, perspective, 1]])

    # 合成单应性矩阵
    # H = T @ R @ S
    H = T @ R @ S @ P

    # 应用单应性变换
    transformed_image = cv2.warpPerspective(image, H, (width, height))

    # convert the black background to white
    transformed_image[transformed_image == 0] = 255

    return transformed_image, H

def find_polygon_vertices(image):

    # convert the image to gray scale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 3)

    # Approximate contours to polygons and get vertices
    epsilon = 0.01 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    vertices = [vertex[0] for vertex in approx]
    # print(vertices)

    # for vertex in vertices:
    #     cv2.circle(image, tuple(vertex), 10, (0, 0, 255), -1)   # 画出顶点
    # # plot the image and the contour
    # plt.imshow(image)
    # plt.show()

    return vertices

def add_shadow_blocks(image, num_blocks=3, shadow_intensity=0.5, min_vertices=4, max_vertices=8):
    """
    在图像中添加随机多边形的阴影块。

    参数:
    - image: 原始图像
    - num_blocks: 要添加的阴影块数量
    - shadow_intensity: 阴影强度，范围从0到1，值越小阴影越深
    - min_vertices: 多边形的最小顶点数
    - max_vertices: 多边形的最大顶点数

    返回:
    - 带有阴影块的图像
    """
    h, w, _ = image.shape
    shadowed_image = image.copy()

    total_area = h * w
    target_shadow_area = total_area / 10  # 总阴影面积约为图像的1/10
    block_area = target_shadow_area / num_blocks

    for _ in range(num_blocks):
        # 随机生成多边形的顶点数
        num_vertices = np.random.randint(min_vertices, max_vertices)
        x_points = np.random.randint(0, w, num_vertices)
        y_points = np.random.randint(0, h, num_vertices)

        rr, cc = polygon(y_points, x_points)

        # 计算并调整多边形的大小以适应目标面积
        poly_area = len(rr)
        resize_factor = np.sqrt(block_area / poly_area)
        rr = (resize_factor * (rr - rr.mean()) + rr.mean()).astype(int)
        cc = (resize_factor * (cc - cc.mean()) + cc.mean()).astype(int)

        # 保证坐标在图像范围内
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)

        # 应用阴影
        shadowed_image[rr, cc] = (shadow_intensity * shadowed_image[rr, cc]).astype(np.uint8)

    return shadowed_image

def calculate_H_with_sift(image1, image2, ratio_thresh=0.75, ransac_thresh=5.0):

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测并计算特征点和描述符
    keypoints_original, descriptors_original = sift.detectAndCompute(image1, None)
    keypoints_rotated, descriptors_rotated = sift.detectAndCompute(image2, None)

    # 使用FLANN匹配器进行特征点匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_original, descriptors_rotated, k=2)

    # 使用 Lowe's ratio 测试筛选良好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 确保有足够的良好匹配点来计算单应性
    if len(good_matches) > 4:

        # 获取匹配点的坐标
        src_pts = np.float32([keypoints_original[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_rotated[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        return H
    
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        return None
    
def calculate_H_with_orb(image1, image2, ratio_thresh=0.75, ransac_thresh=5.0):
    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测并计算特征点和描述符
    keypoints_original, descriptors_original = orb.detectAndCompute(image1, None)
    keypoints_rotated, descriptors_rotated = orb.detectAndCompute(image2, None)

    # 使用FLANN匹配器进行特征点匹配
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_original, descriptors_rotated, k=2)

    # 使用 Lowe's ratio 测试筛选良好的匹配点
    good_matches = []

    if matches is None:
        return None
    
    else:

        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)


    # 确保有足够的良好匹配点来计算单应性
    if len(good_matches) > 4:

        # 获取匹配点的坐标
        src_pts = np.float32([keypoints_original[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_rotated[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        return H
    
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        return None
    
def calculate_H_with_loftr(img1, img2, ratio_thresh=0.75, ransac_thresh=5.0):

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
    mkpts0 = correspondences["keypoints0"]
    mkpts1 = correspondences["keypoints1"]
    confidence = correspondences['confidence']

    # matches number
    matches_count = len(mkpts0)

    # Move tensors to GPU if available
    mkpts0 = mkpts0.to(device)
    mkpts1 = mkpts1.to(device)
    confidence = confidence.to(device)

    # Apply ratio test
    good_matches = mkpts0[confidence > ratio_thresh]
    good_matches_count = good_matches.shape[0]

    # update the mkpts0 and mkpts1 with the good matches
    mkpts0 = mkpts0[confidence > ratio_thresh].cpu().numpy()
    mkpts1 = mkpts1[confidence > ratio_thresh].cpu().numpy()

    # Apply RANSAC to filter out outliers and estimate a homography (if needed)
    if good_matches_count > 4:
        H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_thresh)

        return H
    else:
        print('Not enough matches were found.')
        return None
    
def warp_image(image, H):

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 应用单应性变换
    warped_image = cv2.warpPerspective(image, H, (width, height))

    # convert the black background to white
    warped_image[warped_image == 0] = 255

    return warped_image

def calculate_similarities(random_H, estimated_H):
    # 计算随机生成的单应性矩阵和估计的单应性矩阵之间的相似性

    # 计算两个矩阵之间的Frobenius范数
    frobenius_norm_diff = np.linalg.norm(random_H - estimated_H, 'fro')

    # 计算余弦相似度
    # 首先展平矩阵变成向量
    random_H_flatten = random_H.flatten()
    estimated_H_flatten = estimated_H.flatten()
    # 计算余弦相似度
    cosine_similarity = (np.dot(random_H_flatten, estimated_H_flatten) /
                         (np.linalg.norm(random_H_flatten) * np.linalg.norm(estimated_H_flatten)))

    return frobenius_norm_diff, cosine_similarity


def calculate_image_similarity(image1, image2, alpha=1.0, beta=0.0, ksize=3):
        # calculate the mse, psnr, ssim
        mse_value = mse(image1, image2)
        psnr_value = cv2.PSNR(image1, image2)

        # convert the image to gray scale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(image1_gray, image2_gray)

        return mse_value, psnr_value, ssim_value

def pad_image_to_match(image1, image2):
    # Get dimensions of both images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Calculate padding sizes
    vertical_padding = abs(h1 - h2)
    horizontal_padding = abs(w1 - w2)

    # Pad the smaller image
    if h1 < h2:
        image1 = cv2.copyMakeBorder(image1, 0, vertical_padding, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        image2 = cv2.copyMakeBorder(image2, 0, vertical_padding, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

    if w1 < w2:
        image1 = cv2.copyMakeBorder(image1, 0, 0, 0, horizontal_padding, cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        image2 = cv2.copyMakeBorder(image2, 0, 0, 0, horizontal_padding, cv2.BORDER_CONSTANT, value=[255,255,255])

    return image1, image2


# img_folder_path = r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton\plot\ponder_9A\test\image'
# images = load_images_from_folder(img_folder_path)

# nested_folder_path = r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton_plot\20231018_Ponder_9A_left'

nested_folder_path = '/blue/lift-phenomics/zhengkun.li/peanut_project/image_stitching/data/tifton_plot/20231018_Ponder_9A_right'


# nested_folder_path = r'X:\zhengkun.li\peanut_project\image_stitching\data\tifton_plot\1'
# images = load_images_from_nested_folders(nested_folder_path)


# save the all the image result to a csv file
with open('comparsion_sift-orb-loftr-right-crop-white2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'random_H', 'calculated_H_sift','MSE_sift', 'PSNR_sift', 'SSIM_sift', 'Frobeninus_sift', 'cosine_sift', 'calculated_H_orb', 'MSE_orb', 'PSNR_orb', 'SSIM_orb', 'Frobeninus_orb', 'cosine_orb', 'calculated_H_loftr', 'MSE_loftr', 'PSNR_loftr', 'SSIM_loftr', 'Frobeninus_loftr', 'cosine_loftr'])

    for folder in os.listdir(nested_folder_path):
        folder_path = os.path.join(nested_folder_path, folder)

        print(f'Processing folder: {folder}')
        images = load_images_from_folder(folder_path)

        augned_image = None
        transformed_image = None
        warped_image_loftr = None
        warped_image_sift = None
        warped_image_orb = None

        for i, image in enumerate(images):

            image_org = image.copy()

            height_crop_rate = [0.1, 0.15, 0.2, 0.25, 0.3]
            # crop the image with a random rate
            crop_rate = np.random.choice(height_crop_rate)
            height, width = image.shape[:2]
            
            image[: int(height * crop_rate), :] = [255, 255, 255]

            augned_image = random_variety(image_org)
            augned_image[int(height * (1 - crop_rate)):, :] = [255, 255, 255]
            transformed_image, random_H = random_homography(augned_image)

            transformed_image[int(height * (1 - crop_rate)):, :] = [255, 255, 255]




            calculated_H_sift = calculate_H_with_sift(image, transformed_image)
            calculated_H_orb = calculate_H_with_orb(image, transformed_image)
            calculated_H_loftr = calculate_H_with_loftr(image, transformed_image)

            row_data = [f'{folder}_{i}', random_H]
            if calculated_H_sift is not None:
                warped_image_sift = warp_image(augned_image, calculated_H_sift)
                warped_image_sift[int(height * (1 - crop_rate)):, :]   = [255, 255, 255]
                # warped_image_sift = warp_image(augned_image[int(height * crop_rate):,:], calculated_H_sift)
                # mse_sift, psnr_sift, ssim_sift = calculate_image_similarity(transformed_image, warped_image_sift)

                transformed_image, warped_image_sift = pad_image_to_match(transformed_image, warped_image_sift)
                mse_sift, psnr_sift, ssim_sift = calculate_image_similarity(transformed_image, warped_image_sift)

                frobeninus_sift, cosine_sift = calculate_similarities(random_H, calculated_H_sift)
                row_data.extend([calculated_H_sift, mse_sift, psnr_sift, ssim_sift, frobeninus_sift, cosine_sift])

            else:
                row_data.extend([None, None, None, None, None, None])

            if calculated_H_orb is not None:
                warped_image_orb = warp_image(augned_image, calculated_H_orb)
                warped_image_orb[int(height * (1 - crop_rate)):, :]   = [255, 255, 255]
                # warped_image_orb = warp_image(augned_image[int(height * crop_rate):,:], calculated_H_orb)
                # mse_orb, psnr_orb, ssim_orb = calculate_image_similarity(transformed_image, warped_image_orb)

                transformed_image, warped_image_orb = pad_image_to_match(transformed_image, warped_image_orb)
                mse_orb, psnr_orb, ssim_orb = calculate_image_similarity(transformed_image, warped_image_orb)

                frobeninus_orb, cosine_orb = calculate_similarities(random_H, calculated_H_orb)
                row_data.extend([calculated_H_orb, mse_orb, psnr_orb, ssim_orb, frobeninus_orb, cosine_orb])

            else:    
                row_data.extend([None, None, None, None, None, None])

            if calculated_H_loftr is not None:
                warped_image_loftr = warp_image(augned_image, calculated_H_loftr)
                warped_image_loftr[int(height * (1 - crop_rate)):, :]   = [255, 255, 255]
                # warped_image_loftr = warp_image(augned_image[int(height * crop_rate):,:], calculated_H_loftr)

                transformed_image, warped_image_loftr = pad_image_to_match(transformed_image, warped_image_loftr)
                mse_loftr, psnr_loftr, ssim_loftr = calculate_image_similarity(transformed_image, warped_image_loftr)

                # mse_loftr, psnr_loftr, ssim_loftr = calculate_image_similarity(transformed_image, warped_image_loftr)
                frobeninus_loftr, cosine_loftr = calculate_similarities(random_H, calculated_H_loftr)
                row_data.extend([calculated_H_loftr, mse_loftr, psnr_loftr, ssim_loftr, frobeninus_loftr, cosine_loftr])

            else:
                row_data.extend([None, None, None, None, None, None])

            # plot the image (2.3): image, augned_image, transformed_image, warped_image_sift, warped_image_orb, warped_image_loftr
            # fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            # ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # ax[0, 0].set_title('Original Image')
            # ax[0, 0].axis('off')

            # ax[0, 1].imshow(augned_image)
            # ax[0, 1].set_title('Augmented Image')
            # ax[0, 1].axis('off')

            # ax[0, 2].imshow(transformed_image)
            # ax[0, 2].set_title('Transformed Image')
            # ax[0, 2].axis('off')

            # if warped_image_sift is not None:
            #     ax[1, 0].imshow(warped_image_sift)
            #     ax[1, 0].set_title('Warped Image (SIFT)')
            #     ax[1, 0].axis('off')

            # if warped_image_orb is not None:

            #     ax[1, 1].imshow(warped_image_orb)
            #     ax[1, 1].set_title('Warped Image (ORB)')
            #     ax[1, 1].axis('off')

            # if warped_image_loftr is not None:

            #     ax[1, 2].imshow(warped_image_loftr)
            #     ax[1, 2].set_title('Warped Image (LoFTR)')
            #     ax[1, 2].axis('off')
            
            # # # show the axis value
            # for i in range(2):
            #     for j in range(3):
            #         ax[i, j].axis('on')


            # plt.tight_layout()

            # plt.show()
            # plt.savefig(f'./images/{folder}_{i}.png')

            # plt.close()

            print(row_data)
            writer.writerow(row_data)


