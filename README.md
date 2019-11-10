This project implementation is based on the following paper:

Youbao Tang, Ke Yan*, Yuxing Tang*, Jiamin Liu*, Jing Xiao, Ronald M. Summers, "ULDor: A Universal Lesion Detector for CT Scans with Pseudo Masks and Hard Negative Example Mining," ISBI, 2019 [(arXiv)](https://arxiv.org/abs/1901.06359) 

In addition the data preprocessing steps were adapted from:
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
Several pre-processing steps were conducted on the image and labels prior to serving them to the model. These steps were placed in a data pipeline so that the same pipeline could be used during the training, validation and testing phase.

1. Offset Adjustment: Subtract 32768 from the pixel intensity of the stored unsigned 16 bit images to obtain the original Hounsfield unit (HU) values. The Hounsfield scale is a quantitative scale for describing radiodensity.
2. Intensity Windowing<sup>[4](https://radiopaedia.org/articles/windowing-ct)</sup>: Rescale intensity from range in window to floating-point numbers in [0,255]. Different structures (lung, soft tissue, bone etc.), have
different windows however a for this project a single range (-1024,3071 HU) is used that
covers the intensity ranges of the lung, soft tissue, and bone.
3. Resizing: Resize every image slice so that each pixel corresponds to 0.8mm.
4. Tensor Conversion: Covert image and labels to tensors
5. Clip black border (commented out in code): Clip black borders in image for computational efficiency and adjust labels (bounding box and mask) accordingly. For some unknown reason this transformation is apparently preventing the model from learning useful features and does not allow the training loss to converge. This merits further investigation

PyTorch has tools to streamline the data preparation process used in many machine learning problems. Below I briefly go through the concepts which are used to make data loading easy.

torch.utils.data.Dataset class:

This is an abstract class which represents the dataset. In this project the class DeepLesion is a subclass of the Dataset class. DeepLesion overrides the following methods of the Dataset class:

* \_\_len__ so that len(dataset) returns the size of the dataset.
* \_\_getitem__ to support the indexing such that dataset[i] can be used to get ith sample

A sample of the DeepLesion dataset will be a tuple consisting of the CT scan image (torch.tensor) and a dictionary of labels and meta data. The dictionary has the following structure:

* boxes : List of bounding boxes of each lesion in image
* masks : Instance segmentation mask for each lesion in image
* labels : List of 1's because 1 represents the label of the lesion class
* image_id : String storing the relative filename of image slice (e.g. 004408_01_02\\088.png)

DeepLesion's initializer also takes an optional argument 'transform' which is used to apply the preprocessing steps described above

Transformations: 
For each preprocessing/transformation step a separate class is created. These classes will implement a \_\_call__ method and \_\_init__ method. The \_\_init__ is used to customize the transformation. For example in the ToOriginalHU class 'offset' is passed to the \_\_init__ method. The \_\_call__ method on the other hand receives the parameters which are potentially transformed. As a result ToOriginalHU subtracts the 'offset' value from the image, which is passed as a parameter to the \_\_call__ method. This is what the resulting code looks like:

    class ToOriginalHU(object):
        """Subtracting offset from the16-bit pixel intensities to
        obtain the original Hounsfield Unit (HU) values"""
    
        def __init__(self, offset):
            self.offset = offset
    
        def __call__(self, image, spacing=None, targets=None, ):
            image = image.astype(np.float32, copy=False) - self.offset
            return image, spacing, targets
All such classes are placed together in a list and the resulting list is passed to the Compose class initializer

Compose class:

This class also has a an \_\_init__ method and a \_\_call__ method. The \_\_init__ method initializes the class with a collection of classes each representing a transformation as described above. The \_\_call__ method simply traverses the collection instantiating each transformation and storing the result of each transformation in the same variables which are passed as parameters to the next transformation. By doing this Compose chains the preprocessing steps together.

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms
    
        def __call__(self, image, spacing=None, targets=None):
            for t in self.transforms:
                image, spacing, targets = t(image, spacing, targets)
            return image, spacing, targets
Now let's look at how these concepts are used in the project:

    from data import transforms as T
    data_transforms = {
        'train': T.Compose([T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
        , 'val': T.Compose([T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
        , 'test': T.Compose([T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
    }
    image_datasets = {x: DeepLesion(DIR_IN + os.sep + x, GT_FN_DICT[x], data_transforms[x]) for x in ['train', 'val'
                                                                                                      , 'test']}
The above code snippet first defines a data_transformation dictionary which has 'train', 'val' and 'test' as the key values and an instance of the Compose class with all preprocessing steps as the value for each key. Similarly the image_dataset is a dictionary with the same keys and the values contain an instance of the DeepLesion class. Note that an instance of the Compose class is passed as a parameter to create an instance of the DeepLesion class. This instance of the Compose class is stored in the trasnform variable of the DeepLesion class. 