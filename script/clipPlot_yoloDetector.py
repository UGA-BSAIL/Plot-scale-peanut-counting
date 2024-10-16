# from ultralytics import YOLO
import numpy as np
import cv2
import torch
import os, sys
sys.path.append("/blue/lift-phenomics/zhengkun.li/yolov8")
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt

import pandas as pd,os
from scipy.signal import find_peaks

# video path and YOLO model path
# video_path = "/blue/lift-phenomics/zhengkun.li/peanut_project/data/20230815/peanut_row14_rightview_20230815.MP4"
# model = YOLO('/blue/lift-phenomics/zhengkun.li/peanut_project/training_result/peanut_yolov8x_second_try/train/weights/best.pt')
video_path = "/orange/cli2/zhengkun.li/peanut_project/citra_data/20230815/raw_video/DSLR/cam2/P1060025.MP4"
model = YOLO('/orange/cli2/zhengkun.li/peanut_project/model_training/weight/YOLOv8/best_citra.pt')
save_path = "/orange/cli2/zhengkun.li/peanut_project/citra_data/20230815/video_extraction/DSLR/cam2/P1060025"

# Check if the directory exists, and if not, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get video name and create CSV and plot filenames based on it
video_name = os.path.basename(video_path).split('.')[0]
save_path_csv = os.path.join(save_path, video_name)
csv_filename = f"{save_path_csv}_detection_result.csv"
plot_filename = f"{save_path_csv}_detection_result.png"

# Initialize CSV file
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame_count', 'num_boxes'])  # Write column names

# Open video file
video_capture = cv2.VideoCapture(video_path)

# Target frame rate and size
target_fps = 3
target_scale = 1

frame_count = 0

# Data for plotting
frame_counts = []
num_boxes = []

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(f'{video_path}_detection_result.MP4', fourcc, target_fps, (width, height))

while True:
    success, frame = video_capture.read()
    if not success:
        break
    
    frame_count += 1

    if frame_count % (int(video_capture.get(cv2.CAP_PROP_FPS)) // target_fps) == 0:
        height, width = frame.shape[:2]
        new_height = int(height * target_scale)
        new_width = int(width * target_scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        results = model(resized_frame, conf=0.3, max_det=2000)
        boxes = []

        for result in results:
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            for box in result.boxes.xyxy:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        text = f'Frame num: {frame_count}, detection number: {len(boxes)}'
        print(text)
        cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4)
        out.write(frame)
        # Save data to CSV
        csv_writer.writerow([frame_count, len(boxes)])
        
        # Save data for plotting
        frame_counts.append(frame_count)
        num_boxes.append(len(boxes))

# Release resources
out.release()
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()

# Plotting the graph
plt.figure(figsize=(12, 6))
plt.plot(frame_counts, num_boxes)
plt.xlabel('Frame')
plt.ylabel('Number of Boxes')
plt.title('The distribution of detection among the video')
save_path_distribution = os.path.join(save_path, f"{video_name}_detection_result.png")
plt.savefig(save_path_distribution)  # Save the plot as a file
# plt.show()



df = pd.read_csv(csv_filename)
# soothing the curve
smoothed_num_boxes = df['num_boxes'].rolling(window=3).max()

# find the valleys of the curve
peaks, _ = find_peaks(-smoothed_num_boxes, distance=30)

# only keep the valleys where the smoothed_num_boxes value is less than 20
filtered_peaks = [peak for peak in peaks if smoothed_num_boxes[peak] <= 20]

# plot the smoothed curve and the valleys
plt.figure(figsize=(12, 6))
plt.plot(df['frame_count'], smoothed_num_boxes, label='Smoothed num_boxes')
plt.plot(df['frame_count'][filtered_peaks], smoothed_num_boxes[filtered_peaks], 'ro', label='Valleys')
plt.xlabel('Frame Count')
plt.ylabel('Smoothed num_boxes')
plt.legend()
# plt.show()
plot_filename_smooth =  f"{video_name}_detection_result_smooth.png"
save_path_smooth = os.path.join(save_path, plot_filename_smooth)
plt.savefig(save_path_smooth)  # Save the plot as a file

# plt.savefig(plot_filename_smooth)  # Save the plot as a file
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

video_segments = []
start_frame = 0
start_frame_index = 0

for peak_frame in df['frame_count'][filtered_peaks]:

    end_frame = peak_frame
    end_frame_index = df[df['frame_count'] == end_frame].index.tolist()[0]
    relevant_smoothed_values = smoothed_num_boxes[start_frame_index:end_frame_index]
    
    # check if the maximum value of smoothed_num_boxes in the start_frame to end_frame interval is greater than 20/40
    if relevant_smoothed_values.max() > 20:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
        save_path_video = os.path.join(save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_{start_frame}-{end_frame}.MP4")
        # save_path_video = os.path.join(save_path, f"{video_name}_{start_frame}-{end_frame}.MP4")
        out = cv2.VideoWriter(save_path_video, fourcc, fps, (width, height))
        
        while start_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            start_frame += 1

        out.release()
        print(save_path_video, fourcc, fps, (width, height))
    else:
        save_path_video = os.path.join(save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_{start_frame}-{end_frame}.MP4")
        print(f'{save_path_video}, because not all smoothed_num_boxes values are greater than 40.')

    start_frame_index = end_frame_index
    start_frame = end_frame

cap.release()
cv2.destroyAllWindows()
