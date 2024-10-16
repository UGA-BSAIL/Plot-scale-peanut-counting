# Plot-scale-peanut-counting
The repository is for the paper: Robotic Plot-Scale Peanut Yield Estimation using Transformer-based Image Stitching and Detection

## Pipeline
<p align="center">
  <img src="figures/fig1_diagram_.jpg" alt="Fig. 1: Diagram of the proposed blueberry fruit phenotyping workflow involving four stages: data collection, training dataset generation, model training, and phenotyping traits extraction." style="width: 70%;">
</p>
<p align="center"><i>Fig. 1: Diagram of the proposed blueberry fruit phenotyping workflow involving four stages: data collection, training dataset generation, model training, and phenotyping traits extraction.</i></p>

## SAM-based Pixel-wise labeling
<p align="center">
  <img src="figures/fig4_SAM-based-labeling_.jpg" alt="Figure 2. Illustration of the proposed automated pixel-wise label generation process for blueberry fruits at different maturiety stages. (a) Bounding boxes from a previous detection dataset (Z. Li et al., 2023); (b) Bounding boxes re-classified into three categories: immature (yellow), semi-mature(red), and mature (blue), using a maturity classifier; (c) Pixel-wise mask labels generated using the Segment Anything Model." style="width: 70%;">
</p>
<p align="center"><i>Figure 2. Illustration of the proposed automated pixel-wise label generation process for blueberry fruits at different maturiety stages. (a) Bounding boxes from a previous detection dataset (Z. Li et al., 2023); (b) Bounding boxes re-classified into three categories: immature (yellow), semi-mature(red), and mature (blue), using a maturity classifier; (c) Pixel-wise mask labels generated using the Segment Anything Model.</i></p>

## BerryNet Architecture
<p align="center">
  <img src="figures/fig7_BerryNet-architecture_.jpg" alt="Figure 3. Illustration of the BerryNet framework. It incorporated three major enhancements: 1) enhancing P2 layer to better capture features of small objects; 2) implementing BiFPN for improved feature fusion, and 3) replacing C2f block with the more efficient C2f-faster block to accelerate inference. " style="width: 70%;">
</p>
<p align="center"><i>Figure 3. Illustration of the BerryNet framework. It incorporated three major enhancements: 1) enhancing P2 layer to better capture features of small objects; 2) implementing BiFPN for improved feature fusion, and 3) replacing C2f block with the more efficient C2f-faster block to accelerate inference. </i></p>



## Prerequisites

[YOLOv8](https://github.com/ultralytics/ultralytics)
```
pip install ultralytics
```
[Segment Anything Model](https://github.com/ultralytics/ultralytics)
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```



## Environment Setting
Clone the repository to local machine:
```
git clone https://github.com/UGA-BSAIL/BerryNet.git
```
Create a virtual env and Install the required packages :
```
conda create -n BerryNet python=3.8
conda activate BerryNet
pip install ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
```
We modified the original YOLOv8 repository for more module support (yolov8-BerryNet\ultralytics\nn\extra_modules). For letting ultralytics point to the modified repository, 
```
pip uninstall ultralytics
```

## Dataset Download
This paper released four datasets for comprhensive research of blueberry, which are availiable on kaggle:
  * [Blueberry Fruit Detection](https://www.kaggle.com/datasets/zhengkunli3969/blueberry-detection-dataset)
  * [Blueberry Maturity Classification](https://www.kaggle.com/datasets/zhengkunli3969/blueberry-maturiety-classification)
  * [Blueberry Pixel-wise Segmentation](https://www.kaggle.com/datasets/zhengkunli3969/blueberry-segmentation-with-segment-anything-model)
  * [Blueberry Cluster Detection](https://www.kaggle.com/datasets/zhengkunli3969/blueberry-cluster-detection)


## SAM-based Pixel-wise labeling
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
BerryNet's architecture was defined in yolov8-BerryNet\ultralytics\models\v8\yolov8-c2f_faster-p2-bifpn-seg.yaml.
For training the model, run the script of train_blueberry_c2f-faster_p2_bifpn_seg_640.py (larger model select the 1280) under the path of yolov8-BerryNet folder:
```
cd yolov8-BerryNet
python train_blueberry_c2f-faster_p2_bifpn_seg_640.py
```
Before running the script, please modify the path of the dataset and the model configuration file in the script. 
You can try more yaml files for different model architecture.


## Pretrained models
The pre-trained models are available at [weight](weight).  
    &nbsp; &nbsp;- Maturity_classifier:  
    &nbsp; &nbsp;- Segment Anything Model:  
    &nbsp; &nbsp;- Cluster Detection:  
    &nbsp; &nbsp;- Fruit Segmentaiton:  


## Model Inference
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
