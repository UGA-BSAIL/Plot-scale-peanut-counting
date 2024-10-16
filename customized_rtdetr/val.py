import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR, YOLO

if __name__ == '__main__':
    model = RTDETR('/blue/lift-phenomics/zhengkun.li/peanut_project/training_result/peanut_rtdetr-resnet18-FasterBlock-ADown-Dysample-r18-640/peanut_rtdetr-resnet18-FasterBlock-ADown-Dysample-epoch1000/weights/best.pt')
    # model = RTDETR('/blue/lift-phenomics/zhengkun.li/peanut_project/training_result/yolov8x-640/exp/weights/best.pt')
    
    # model  = YOLO('/blue/lift-phenomics/zhengkun.li/peanut_project/training_result/yolov8x-640/exp/weights/best.pt')
    model.val(data='/blue/lift-phenomics/zhengkun.li/peanut_project/data/peanut_tifton_patch/data.yaml',
              split='test',
              imgsz=640,
              batch=1,
              max_det = 3000,
              iou = 0.5,
              conf = 0.3,
              save_json=True, # if you need to cal coco metrice
              save_txt = True,
              save = True,
              project='runs/val',
              # name='exp',
              )