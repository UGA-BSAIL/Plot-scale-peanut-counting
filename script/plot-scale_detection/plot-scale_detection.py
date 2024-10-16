from ultralytics import YOLO
import os
import pandas as pd
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PIL import Image, ImageEnhance
Image.MAX_IMAGE_PIXELS = None
import cv2 as cv
# Load a pretrained YOLOv8n model

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='/blue/lift-phenomics/zhengkun.li/peanut_project/Analysis_result/SFM_MetaShape/YOLOv8/best.pt',
    # config_path=model_config_path,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
    # image_size = image_size,
)

image_folder = '/blue/cli2/zhengkun.li/peanut_project/Analysis_result/LoFTR-stitching/stitching-result/loftr-y-translation/third/left'
# model = YOLO('best.pt')
# Get the list of files with the specified extension in the folder
image_files = [file for file in os.listdir(image_folder) if file.endswith(f'.{"jpg"}')]

objects = {}
# Print the list of files
for file in image_files:
    image_path = os.path.join(image_folder, file)
    print(f'processing image: {image_path}')
    result = get_sliced_prediction(
        image_path,
        detection_model,
        # slice_height=2560,
        # slice_width=2560,
        slice_height=1560,
        slice_width=1560,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        verbose = 2
    )

    objects[file] = len(result.object_prediction_list)

    logging.info(file+" found "+str(len(result.object_prediction_list))+" objects.")

    result.export_visuals(export_dir="/blue/cli2/zhengkun.li/peanut_project/Analysis_result/LoFTR-stitching/detection_result/loftr-y-translation/third/left", file_name=file, hide_conf=True, hide_labels=True, rect_th=2)
logging.info(objects)

# save the detection results to an excel file
df = pd.DataFrame(list(objects.items()), columns=['Filename', 'Object_Count'])
excel_file_path = '/blue/cli2/zhengkun.li/peanut_project/Analysis_result/LoFTR-stitching/detection_result/loftr-y-translation/third/left/detection_results_SFM.xlsx'
df.to_excel(excel_file_path, index=False)