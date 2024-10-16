# Plot-scale-peanut-counting
The repository is for the paper: Robotic Plot-Scale Peanut Yield Estimation using Transformer-based Image Stitching and Detection

## Pipeline
<p align="center">
  <img src="figures/figure1-workflow.png" alt="Fig. 1: Workflow of automated plot-scale peanut pod counting using image stitching and deep learning detection." style="width: 70%;">
</p>
<p align="center"><i>Fig. 1: Diagram of the proposed blueberry fruit phenotyping workflow involving four stages: data collection, training dataset generation, model training, and phenotyping traits extraction.</i></p>

## LoFTR-based image stitching
<p align="center">
  <img src="figures/figure2-stitching.png" alt="Figure 2. The procedure of the image stitching algorithm using LoFTR. " style="width: 70%;">
</p>
<p align="center"><i>Fig. 2: The procedure of the image stitching algorithm using LoFTR. </i></p>

## Customized RT-DETR Architecture
<p align="center">
  <img src="figures/figure3-customized-rtdetr.png" alt="Illustration of improved RT-DETR detector. (a) overview of customized RT-DETR detector; (b) Backbone of ResNet18-FasterBlock; (c) Up sampling based on DySample; (d) Adown module for down sampling." style="width: 70%;">
</p>
<p align="center"><i>Fig. 3: Illustration of improved RT-DETR detector. (a) overview of customized RT-DETR detector; (b) Backbone of ResNet18-FasterBlock; (c) Up sampling based on DySample; (d) Adown module for down sampling. </i></p>

## Result example
<p align="center">
  <img src="figures/figure4-result.png" alt="Figure 4.Illustration of plot-scale pod counting" style="width: 70%;">
</p>
<p align="center"><i>Fig. 4: Illustration of plot-scale pod counting. </i></p>


## Prerequisites

[YOLOv8](https://github.com/ultralytics/ultralytics)
```
pip install ultralytics
```

## Environment Setting
Clone the repository to local machine:
```
git clone https://github.com/UGA-BSAIL/Plot-scale-peanut-counting.git
```
Create a virtual env and Install the required packages :
```
conda create -n rt-detr-peanut python=3.8
conda activate rt-detr-peanut
pip install ultralytics
```
We modified the original YOLOv8 repository for more module support (yolov8-BerryNet\ultralytics\nn\extra_modules). For letting ultralytics point to the modified repository, 
```
pip uninstall ultralytics
```


## Dataset Download
This paper released a dataset for model training and validation of peanut detection, which is available on kaggle:
  * [MARS-Peanut-Detection](https://www.kaggle.com/datasets/zhengkunli3969/mars-peanut-detection)


## LoFTR-based image stitching
This method requred the detection dataset as the initial prompt, and weight of maturity classifier and SAM for inference.
MOdified the path of the dataset and the model in the script of SAM-based-labeling.py:
```
python script\pixle-wise_labeling.py
```
  &nbsp;Parameters:  
    &nbsp; &nbsp;- image_folder = '/path/to/image_folder'  
    &nbsp; &nbsp;- annotation_folder = '/path/to/annotation_folder'  
    &nbsp; &nbsp;- checkpoint = torch.load('/path/to/best_model.pth', map_location=torch.device('cpu'))  # path to the maturity classifier  
    &nbsp; &nbsp;- sam_checkpoint = "/path/to/sam_vit_h_4b8939.pth"  # path to the SAM model  


## Model Training
The model architecture of customized RT-RTDETR was defined in customized_rtdetr/ultralytics/cfg/models/rt-detr/rtdetr-resnet18-FasterBlock-ADown-Dysample.yaml.
For training the model, run the script: 
    -  train-detr-r18-fasterBlock-ADown-Dysample-peanut-1280.py (or select the 640) under the path of customized_rtdetr folder:
```
cd customized_rtdetr
python train-detr-r18-fasterBlock-ADown-Dysample-peanut-1280.py
```
Before running the script, please modify the path of the dataset and the model configuration file in the script. 
You can try more yaml files for different model architecture.


## Pre-trained models
The pre-trained models are available at [weight](weight).  
    &nbsp; &nbsp;- Maturity_classifier:  
    &nbsp; &nbsp;- Segment Anything Model:  


## Inference on plot-scale image
For model inference, run the script of BerryNet_phenotyping_extraction_split.py under the script folder:
```
python script\BerryNet_phenotyping_extraction_split.py
```
  &nbsp;Parameters:  
    &nbsp; &nbsp;- model_path = " "    # path to the BerryNet model  
    &nbsp; &nbsp;- image_folder = " "  # path to the image folder  
    &nbsp; &nbsp;- save_path = " "     # path to the save folder  


## References
If you find this work or code useful, please cite:

```
****************************
```
