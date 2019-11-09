This project implementation is based on the following paper:

Youbao Tang, Ke Yan*, Yuxing Tang*, Jiamin Liu*, Jing Xiao, Ronald M. Summers, "ULDor: A Universal Lesion Detector for CT Scans with Pseudo Masks and Hard Negative Example Mining," ISBI, 2019 [(arXiv)](https://arxiv.org/abs/1901.06359) 

In addition the data postprocessing steps were adapted from:
https://github.com/rsummers11/CADLab/tree/master/lesion_detector_3DCE

The above link is a code implementation of the following paper:
Ke Yan et al., "3D Context Enhanced Region-based Convolutional Neural Network for End-to-End Lesion Detection," MICCAI, 2018 [(arXiv)](https://arxiv.org/abs/1806.09648)  

# UNIVERSAL LESION DETECTOR FOR CT SCANS WITH PSEUDO MASKS

## Introduction:
Image recognition and deep learning technologies using Convolutional Neural Networks (CNN) have demonstrated remarkable progress in the medical image analysis field. Traditionally radiologists with extensive clinical expertise visually asses medical images to detect and classify diseases. The task of lesion detection is particularly challenging because non-lesions and true lesions
can appear similar. 

For my capstone project I use a Mask R-CNN <sup>[1](https://arxiv.org/abs/1703.06870)</sup> to detect lesions in a CT scan. The model outputs a bounding box, instance segmentation and score for each detected lesion. Mask R-CNN was built by the Facebook AI research team (FAIR) in April 2017.

The algorithms are implemented using Pytoch and run on an Nvidia Quadro P4000 GPU. 
## Data:
The dataset used to train the model has a variety of lesion types such as lung nodules, liver tumors and enlarged lymph nodes. This large-scale dataset of CT images, named DeepLesion, is publicly available and has over 32,000 annotated lesions.<sup>[2](https://nihcc.app.box.com/v/DeepLesion/)</sup> The data consists of 32,120 axial computed tomography (CT) slices with 1 to 3 lesions in each image. The annotations and meta-data are stored in three excel files:

DL_info_train.csv (training set)

DL_info_val.csv (validation set)

DL_info_test.csv (test set)

Each row contains information for one lesion. For a list of meanings for each column in the annotation excel files go to: 
https://nihcc.app.box.com/v/DeepLesion/file/306056134060 

Here is a description of some of the key fields:

column 6: Image coordinates (in pixel) of the two RECIST diameters of the lesion. There are 8 coordinates for each annotated lesion and the first 4 coordinates are for the long axis. "Each RECIST-diameter bookmark consists of two lines: one measuring the longest diameter of the lesion and the second measuring its longest perpendicular diameter in the plane of measurement."<sup>[3](https://nihcc.app.box.com/v/DeepLesion/file/306049009356)</sup> These coordinates are used to construct a pseudo-mask for each lesion. More details on this later. 

column 7: Bounding box coordinates which consists of the upper left and lower right coordinates of the bounding box for each annotated lesion.

column 13: Distance between image pixels in mm for x,y,z axes. The third value represents the vertical distance between image slices 

An important point to note is that the total size of the images in the dataset is 225GB however out of the 225GB there is only annotation (i.e. labelled) information for images totaling 7.2GB in size. In this implementation training was only done on a portion of the labelled data.
## Image Pre-Processing and Data Pipeline:
Several image pre-processing steps were conducted prior to serving the images to the model. These steps were placed in a data pipeline so that the same pipeline could be used during the training, validation and testing phase.

1. Offset Adjustment: Subtract 32768 from the pixel intensity of the stored unsigned 16 bit images to obtain the original Hounsfield unit (HU) values. The Hounsfield scale is a quantitative scale for describing radiodensity.
2. Intensity Windowing<sup>[4](https://radiopaedia.org/articles/windowing-ct)</sup>: Rescale intensity from range in window to floating-point numbers in [0,255]. Different structures (lung, soft tissue, bone etc.), have
different windows however a for this project a single range (-1024,3071 HU) is used that
covers the intensity ranges of the lung, soft tissue, and bone.
3. Resizing: Resize every image slice so that each pixel corresponds to 0.8mm.
4. Tensor Conversion: Covert image and labels to tensors

PyTorch has tools to streamline the data preparation process used in many machine learning problems. Below I briefly go through the classes which are used to make data loading easy.

Dataset Class:
This is an abstract class which represents the dataset. In this project the class DeepLesion is a subclass of the Dataset class. DeepLesion overrides the following methods of the Dataset class:

* \_\_len__ so that len(dataset) returns the size of the dataset.
* \_\_getitem__ to support the indexing such that dataset[i] can be used to get ith sample

A sample of the DeepLesion dataset will be a tuple consisting of the CT scan image (torch.tensor) and a dictionary of labels and meta data. The dictionary has the following structure:

* boxes : List of bounding boxes of each lesion in image
* masks : Instance segmentation mask for each lesion in image
* labels : This is a list of 1's because 1 represents the label of the lesion class
* image_id : String storing the relative filename of image (e.g. 004408_01_02\\088.png)    