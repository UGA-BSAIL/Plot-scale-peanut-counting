import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-resnet18-SENetV2-ADown-Dysample.yaml")
    # model.load('/blue/lift-phenomics/zhengkun.li/update/RTDETR-20240120/RTDETR-main/rtdetr-x.pt') # loading pretrain weights
    model.train(data='/blue/lift-phenomics/zhengkun.li/peanut_project/data/peanut_tifton_patch/data.yaml',
                # data='/blue/lift-phenomics/zhengkun.li/peanut_project/data/peanut_v3/data.yaml',
                cache=False,
                imgsz=640,
                epochs=1000,
                batch=8,
                workers=2,
                device='0',
                resume='/blue/lift-phenomics/zhengkun.li/peanut_project/training_result/peanut_rtdetr-SENetV2-ADown-Dysample-r18-640/peanut_rtdetr-r18-SENetV2-ADown-Dysample-640-epoch1000/weights/last.pt', # last.pt path
                project='/blue/lift-phenomics/zhengkun.li/peanut_project/training_result/peanut_rtdetr-SENetV2-ADown-Dysample-r18-640',
                name='peanut_rtdetr-r18-SENetV2-ADown-Dysample-640-epoch1000',
                )