import time
import logging
from torch.optim import lr_scheduler, Adam, SGD
from data import transforms as T
from data.collate_batch import BatchCollator
from torch.utils.data import DataLoader
from data.datasets.deeplesion import *
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from train_eval import train_one_epoc, evaluate

# A single windowing (−1024 to 3071 HU) that covers the
# intensity ranges of lung, soft tissue, and bone.
# For more information on Intensity Windowing go to
# https://radiopaedia.org/articles/windowing-ct
WINDOWING = [-1024, 3071]

BG_INTENSITY = 32000
INTENSITY_OFFSET = 32768
NORM_SPACING = 0.8  # Resize every image slice so that each pixel corresponds to 0.8mm
MAX_SIZE = 512
GT_FN_TRAIN = 'DL_info_train.csv'  # Ground truth file for training data
GT_FN_VAL = 'DL_info_val.csv'  # Ground truth file for validation data
GT_FN_TEST = 'DL_info_test.csv'  # Ground truth file for test data
GT_FN_DICT = {"train": GT_FN_TRAIN, "val": GT_FN_VAL, "test": GT_FN_TEST}
DIR_IN = 'Images_png'  # input directory


def get_model(pre_trained, pretrained_backbone, numclasses):
    if pre_trained:
        dl_model = maskrcnn_resnet50_fpn(pretrained=pre_trained)

        for param in dl_model.parameters():
            param.requires_grad = False

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = numclasses  # 1 class (lesion) + background

        # get number of input features for the classifier
        in_features = dl_model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        dl_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = dl_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 64
        # and replace the mask predictor with a new one
        dl_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                              hidden_layer,
                                                              num_classes)
    else:
        dl_model = maskrcnn_resnet50_fpn(num_classes=numclasses, pretrained_backbone=pretrained_backbone)
    return dl_model


def main():
    logging.basicConfig(filename='example.log', level=logging.DEBUG)
    data_transforms = {
        'train': T.Compose([T.ClipBlackBorder(BG_INTENSITY), T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
        , 'val': T.Compose([T.ClipBlackBorder(BG_INTENSITY), T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
        , 'test': T.Compose([T.ClipBlackBorder(BG_INTENSITY), T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
    }

    logging.info('Loading data sets')
    image_datasets = {x: DeepLesion(DIR_IN + os.sep + x, GT_FN_DICT[x], data_transforms[x]) for x in ['train', 'val'
                                                                                                      , 'test']}
    logging.info('data sets loaded')
    logging.info('Loading data loaders')
    dl_dataloaders = {x: DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=0
                                    , collate_fn=BatchCollator) for x in ['train', 'val', 'test']}

    logging.info('data loaders loaded\n')
    dl_dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # for batch_id, (inputs, targets) in enumerate(dl_dataloaders['train']):
    #     i = 0
    #     for i, (image, target) in enumerate(zip(inputs, targets)):
    #         img_copy = image.squeeze().numpy()
    #         images = [img_copy] * 3
    #         images = [im.astype(float) for im in images]
    #         img_copy = cv2.merge(images)
    #         for j, (bbox, pseudo_mask) in enumerate(zip(target["boxes"], target["masks"])):
    #             bbox = target["boxes"][j].squeeze().numpy()
    #             bbox = np.int16(bbox)
    #             mask = target["masks"][j].squeeze().numpy()
    #             cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
    #             print(mask.shape)
    #             print(img_copy.shape)
    #             msk_idx = np.where(mask == 1)
    #             img_copy[msk_idx[0], msk_idx[1], 0] = 255
    #         cv2.imshow(str(batch_id) + " " + str(i), img_copy)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dl_model = get_model(True, True, 2)

    params = [p for p in dl_model.parameters() if p.requires_grad]

    # Observe that all parameters are being optimized
    optimizer_ft = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
    # optimizer_ft = Adam(params, lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    num_epochs = 4
    since = time.time()
    # best_model_wts = copy.deepcopy(dl_model.state_dict())

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        train_one_epoc(dl_model, optimizer_ft, exp_lr_scheduler, dl_dataloaders['train'], dl_dataset_sizes['train'])

        outputs = evaluate(dl_model, dl_dataloaders['val'])

        logging.info('Average LLF: {}'.format(outputs[0]))
        logging.info('Average NLF: {}'.format(outputs[1]))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    main()
