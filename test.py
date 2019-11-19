from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection import maskrcnn_resnet50_fpn
from data.datasets.deeplesion import *
import torch
from data import transforms as T
from data.collate_batch import BatchCollator
from torch.utils.data import DataLoader
import logging
from train_eval import evaluate
import time

WINDOWING = [-1024, 3071]
BG_INTENSITY = 32000
INTENSITY_OFFSET = 32768
NORM_SPACING = 0.8  # Resize every image slice so that each pixel corresponds to 0.8mm
MAX_SIZE = 512

DIR_IN = 'Images_png'  # input directory
GT_FN_TRAIN = 'annotation_info' + os.sep + 'DL_info_train_sample.csv'  # Ground truth file for training data
GT_FN_VAL = 'annotation_info' + os.sep + 'DL_info_val_sample.csv'  # Ground truth file for validation data
GT_FN_TEST = 'annotation_info' + os.sep + 'DL_info_test_sample.csv'  # Ground truth file for test data
GT_FN_DICT = {"train": GT_FN_TRAIN, "val": GT_FN_VAL, "test": GT_FN_TEST}
DIR_IN = 'Images_png'  # input directory


def main():
    anchor_generator = AnchorGenerator(
        sizes=tuple([(16, 24, 32, 48, 96) for _ in range(5)]),
        aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))
    rpnhead = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    model = maskrcnn_resnet50_fpn(num_classes=2, pretrained_backbone=True
                                  , max_size=MAX_SIZE, rpn_head=rpnhead
                                  , rpn_anchor_generator=anchor_generator, rpn_pre_nms_top_n_train=12000
                                  , rpn_pre_nms_top_n_test=6000, rpn_post_nms_top_n_train=2000
                                  , rpn_post_nms_top_n_test=300, rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3
                                  , rpn_positive_fraction=0.7, bbox_reg_weights=(1.0, 1.0, 1.0, 1.0)
                                  , box_batch_size_per_image=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=3, shuffle=True, num_workers=0
                                 , collate_fn=BatchCollator) for x in ['train', 'val', 'test']}

    num_epochs = 10
    logging.basicConfig(filename='logs' + os.sep + 'test.log', level=logging.DEBUG)
    since = time.time()
    for epoch in range(num_epochs):
        model.load_state_dict(torch.load('saved_models' + os.sep + str(epoch) + '_deeplesion.pth', map_location=device))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 20)

        llf, nlf = evaluate(model, dataloaders['test'])

        logging.info('LLF: {}'.format(llf))
        logging.info('NLF: {}'.format(nlf) + '\n')

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    main()