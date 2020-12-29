## 0. Eplane Algos - Usage

* The two algos - cash flow minimization and shortest path - can be run from eplane_Algos.py
* The script takes as arguments - 

1. “--task” : “flow” , “path” or “monster” - for cashflow, shortest path or maximum number of monster.
2. “--input” : This is only for "flow" and "path" function - Input format is the same for both the functions. The input relation is a list of lists in format [giver/parent_node, taker/child_node, value]. Example for cash flow - ['p0','p1',1], ['p0','p2',2], ['p1','p2',5], ['p1','p3',1]. While passing this in command line, pass this as a **single string** - "['p0','p1',1], ['p0','p2',2], ['p1','p2',5], ['p1','p3',1]" **or** "['a','b',4], ['a','j',1], ['a','d',3], ['a','c',1], ['b','e',1], ['d','e',8], ['d','g',9], ['d','f',8], ['c','d',1], ['c','j',1], ['c','g',7], ['g','j',2]"

3. “--origin” : Applicable only for the function shortest path. PLEASE ENSURE origin IS CONNECTED TO ALL OTHER NODES ELSE THIS WILL THROW AN ERROR. 
4. “--kids_power” : Applicable only for the function monster. This input is a list of powers of kids guns. Example - '[2,12,4,22]'"
5. “--num_monsters” : Applicable only for the function monster. This is the number of monsters awaiting the kids.

***Examples***

* Running cash flow function -
```
python3 eplane_Algos.py --task flow --input "['p0','p1',1], ['p0','p2',2], ['p1','p2',5], ['p1','p3',1], ['p2','p4',2], ['p2','p5',3], ['p5','p4',2], ['p2','p3',3]"
```
* Output
```
Number of times through all persons: 3

Before optimization stats - 
----------------------------- 
Total value flow before optimization: 19
Net values before optimization - 
p0: -3
p1: -5
p2: -1
p3: 4
p4: 4
p5: 1

******

After optimization stats - 
----------------------------- 
Total value flow after optimization: 9
Net values after optimization - 
p0: -3
p1: -5
p2: -1
p3: 4
p4: 4
p5: 1

******

Value flow decorated: 
--------------------- 
step 1: p1 gives 3 to p3
step 2: p1 gives 1 to p5
step 3: p0 gives 1 to p3
step 4: p0 gives 2 to p4
step 5: p2 gives 1 to p4
step 6: p1 gives 1 to p4
```

* Running shortest path function -
```
python3 eplane_Algos.py --task path --origin a --input "['a','b',4], ['a','j',1], ['a','d',3], ['a','c',1], ['b','e',1], ['d','e',8], ['d','g',9], ['d','f',8], ['c','d',1], ['c','j',1], ['c','g',7], ['g','j',2]"
```
* Output
```
{'j': ['a', 'j'], 'c': ['a', 'c'], 'd': ['a', 'c', 'd'], 'b': ['a', 'b'], 'e': ['a', 'b', 'e'], 'g': ['a', 'c', 'g'], 'f': ['a', 'c', 'd', 'f']}
```

* Running maximum monster function -
```
python3 eplane_Algos.py --task monster --kids_power "[9,6,8]" --num_monsters 11
```
* Output
```
3
```


## 1. Eplane AI Projects - Usage

* The two AI projects - facemask detection & semantic segmentation - can be run from eplane_ai.py.
* The script takes as arguments - 

1. “--task” : “facemask” or “seg” 
2. “--input_mode” : “image” or “video”
3. “--file_url” : The url of the image or video to be processed

***Examples***
- `python3 eplane.py --task facemask --input_mode video --file_url /Users/venkateshmadhava/Documents/eplane/projects/misc_images_videos/face_mask_video_480.mov>`
- `python3 eplane.py --task seg --input_mode image --file_url /Users/venkateshmadhava/Documents/eplane/projects/misc_images_videos/154.jpg`

* Face mask prediction on images - 

![Image of Facemask On Image](https://github.com/ven-madhava/eplane/blob/master/for_documentation/facemask_pred_pred.png)

* Face mask prediction on videos - 

![Image of Facemask On Video](https://github.com/ven-madhava/eplane/blob/master/for_documentation/facemask_video_ok_notok.png)

* Street segmentation prediction on images - 

![Image of Street Seg On Image](https://github.com/ven-madhava/eplane/blob/master/for_documentation/seg_model_example.png)

* Street segmentation prediction on videos - 

![Image of Street Seg On Video](https://github.com/ven-madhava/eplane/blob/master/for_documentation/seg_video.png)


## 2. Eplane Projects - package requirements

* List of dependencies to have on `Python 3.8.2`. This is an exhaustive list which includes support for model training as well.
```
Package           Version 
----------------- --------
appnope           0.1.2   
backcall          0.2.0   
cycler            0.10.0  
dataclasses       0.6     
decorator         4.4.2   
future            0.18.2  
ipykernel         5.3.4   
ipython           7.19.0  
ipython-genutils  0.2.0   
jedi              0.17.2  
jupyter-client    6.1.7   
jupyter-core      4.7.0   
kiwisolver        1.3.1   
matplotlib        3.3.3   
numpy             1.19.4  
opencv-python     4.4.0.46
parso             0.7.1   
pexpect           4.8.0   
pickleshare       0.7.5   
Pillow            8.0.1   
pip               19.2.3  
prompt-toolkit    3.0.8   
ptyprocess        0.6.0   
Pygments          2.7.3   
pyparsing         2.4.7   
python-dateutil   2.8.1   
pyzmq             20.0.0  
setuptools        41.2.0  
six               1.15.0  
torch             1.7.0   
torchvision       0.8.1   
tornado           6.1     
traitlets         5.0.5   
typing-extensions 3.7.4.3 
wcwidth           0.2.5
```
  
## 3. Eplane Projects - Writeup explaining approach, assumptions, thoughts, choice and explanation of the algorithms.

#### 3.1 Face Mask Detection

* **Core Problem Statement** - Given a single image or a video instance, identify people not wearing a facemask.
* **Datasets Used** - The dataset used was made up of ~1000 faces with masks and ~1100 faces without masks.

* **Technical Approach** - The input is a single image (either user uploaded or extracted as a frame from a video) which is processed in the following manner.

1. Identification of faces - Human faces are first identified using pretrained models from opencv. The problem of identifying human faces has already been solved concretely   and hence there’s no need to reinvent the wheel here. 
2. Extraction of faces - Identified faces are then extracted, resized and fed into a custom trained face mask detection model.
3. Face Mask Detection - 
  - A  simple fully convolutional classifier with 2 output nodes (one each for faces with and without a mask). 
  - The decision to treat this as a multiclass classification problem rather than a single probability (probability of face with mask) prediction problem, is because of better      visualisation that is possible on multiclass models.   
  - Training pipeline was made up of image processing, image augmentation, training and visualisation. 
  - Image processing - The input to the model was grayscale face images from the dataset.
  - Image augmentation - Salt & pepper noise & random image flip was utilised for better generalisation. Attached images for reference.
  - Visualisation - Using a CNN visualisation technique called CAM (class activation maps), heatmaps were overlaid on train and test data to ensure that the classifier was           building its dependencies on the “right” features of the image. Attached images for reference.
  - An F1 score of ~90% was achieved on an 80:20 split test set.
4. Inference - The faces extracted from the original user input image are appropriately preprocessed and fed into the trained classifier for identifying faces without a mask.

* Facemask classifier image augmentation example - 

![Image of Street Seg On Video](https://github.com/ven-madhava/eplane/blob/master/for_documentation/facemask_img_aug.png)

* Facemask classifier heat maps with predictions - 

![Image of Street Seg On Video](https://github.com/ven-madhava/eplane/blob/master/for_documentation/facemask_dl_model_heatmap.png)

* Classifier Model
```
simple_classifier(
  (image_encoder): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace=True)
    (3): Dropout2d(p=0.2, inplace=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): LeakyReLU(negative_slope=0.2, inplace=True)
    (7): Dropout2d(p=0.2, inplace=False)
    (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Dropout2d(p=0.2, inplace=False)
    (12): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
    (13): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): LeakyReLU(negative_slope=0.2, inplace=True)
    (15): Dropout2d(p=0.2, inplace=False)
    (16): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
    (17): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (18): LeakyReLU(negative_slope=0.2, inplace=True)
    (19): Dropout2d(p=0.2, inplace=False)
    (20): Conv2d(512, 2, kernel_size=(3, 3), stride=(2, 2))
    (21): Sigmoid()
  )
)
```


#### 3.2 UAV Perception Engine

* **Core Problem Statement** - Identify and localize different obstacles which will then be used for path planning.

- Among object detection, semantic segmentation and instance segmentation, semantic segmentation is best suited for this problem. 

  - vs object detection - Since precise pixel wise classification is required for obstacle detection and  path planning, the standard object detection algorithm is not suitable.
  - vs instance segmentation - Instance level differentiation within a single frame is NOT required for the purpose of obstacle detection and path planning. For example, it is enough to differentiate between cars and streets at pixel level. There is no need to differentiate between one car and another as it serves no purpose for our objective.
  
* **Dataset Used** - An outdoor street view dataset is used for this problem since it resonates with the environment an UAV will need to handle. The specific dataset used is BDD100K.
- The dataset has 7k training image triplets (images, label_map & color_segmented_map) and 1k validation image triplets.
- As per sources, the ~20 semantic classes covered are - 
  - banner
  - billboard
  - lane divider
  - parking sign
  - pole
  - polegroup
  - street light
  - traffic cone
  - traffic device
  - traffic light
  - traffic sign
  - sign frame
  - person
  - rider
  - bicycle
  - bus
  - car
  - caravan
  - motorcycle
  - trailer
  - train
  - truck
  
* **Technical Approach** - The input is a single image (either user uploaded or extracted as a frame from a video) which is processed in the following manner.

1. The core objective of the training is to classify every pixel of an input image. 
2. The training pipeline consists of image augmentation, training, model output to segmentation colormap conversion.
3. About the model - 
  - A fully convolutional UNET encoder-decoder model is used for this problem. Among many other complicated algorithms, this was found to be a relatively simpler approach more       conducive to intuitive understanding & effective debugging.
  - The input to this model is an image of size h,w,3 and the output is a classification map of size h,w,number_classes.
  - Each channel of the model output corresponds to a class of objects such as sky, road etc.
  - Each value in the classification map is a 1 or 0 depending on whether the corresponding pixel in the input image belongs to that particular class of objects.
  - The model is trained to optimise on producing the most appropriate classification maps given an input image.
4. Image augmentation - For better generalisation, random gaussian noise and dropouts are used as augmentation techniques while training. Much more diverse segmentation augmentation would add to better generalization, but had to miss it out in the interest of time.
5. Training - This is a standard UNET training process with the objective of predicting classification maps given an image. The targets used while training, were created using custom functions.
6. Model output to color segmentation map - A custom static function converts classification maps into color coded segmentation map which will be overlaid on the input image for during inference.

* UNET FCN Segmentation Model - 
```
fcn_UNET_segmentation(
  (cl00): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (cl0): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (cl1): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (cl2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (cl3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (cl4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (cl5): Sequential(
    (0): Conv2d(512, 128, kernel_size=(3, 3), stride=(2, 2))
    (1): LeakyReLU(negative_slope=0.2)
    (2): Dropout2d(p=0.1, inplace=False)
  )
  (ul1): Sequential(
    (0): ConvTranspose2d(128, 512, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (ul2): Sequential(
    (0): ConvTranspose2d(1024, 256, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (ul3): Sequential(
    (0): ConvTranspose2d(512, 128, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (ul4): Sequential(
    (0): ConvTranspose2d(256, 64, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (ul5): Sequential(
    (0): ConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (ul6): Sequential(
    (0): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Dropout2d(p=0.1, inplace=False)
  )
  (ul7): Sequential(
    (0): ConvTranspose2d(64, 22, kernel_size=(3, 3), stride=(2, 2))
    (1): Sigmoid()
  )
)
```

* Street segmentation image augmentation example - 

![Image of Street Seg On Video](https://github.com/ven-madhava/eplane/blob/master/for_documentation/seg_image_aug.png)


### side notes

1. Slow Video Playback - A threaded function is required to fix video playback speed in opencv. Couldn't get around to it within specified time.
2. Pixel Perfect Accuracy - In the segmentation task, pixel perfect accuracy is not achieved yet. However this is possible with further hyper-param iterations & model structures.
3. I have also added my model training notebook under training_notebooks for your reference.
