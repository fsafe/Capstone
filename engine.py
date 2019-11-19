import time
import logging
from torch.optim import lr_scheduler, Adam, SGD
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
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
GT_FN_TRAIN = 'annotation_info' + os.sep + 'DL_info_train.csv'  # Ground truth file for training data
GT_FN_VAL = 'annotation_info' + os.sep + 'DL_info_val.csv'  # Ground truth file for validation data
GT_FN_TEST = 'annotation_info' + os.sep + 'DL_info_test.csv'  # Ground truth file for test data
GT_FN_DICT = {"train": GT_FN_TRAIN, "val": GT_FN_VAL, "test": GT_FN_TEST}
DIR_IN = 'Images_png'  # input directory


def get_model(pre_trained, pretrained_backbone, numclasses):
    anchor_generator = AnchorGenerator(
        sizes=tuple([(16, 24, 32, 48, 96) for _ in range(5)]),
        aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))
    rpnhead = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    if pre_trained:
        # dl_model = maskrcnn_resnet50_fpn(pretrained=pre_trained, max_size=MAX_SIZE, rpn_head=rpnhead
        #                                  , rpn_anchor_generator=anchor_generator, rpn_pre_nms_top_n_train=12000
        #                                  , rpn_pre_nms_top_n_test=6000, rpn_post_nms_top_n_train=2000
        #                                  , rpn_post_nms_top_n_test=300, rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3
        #                                  , rpn_positive_fraction=0.7, bbox_reg_weights=(1.0, 1.0, 1.0, 1.0)
        #                                  , box_batch_size_per_image=32)
        dl_model = maskrcnn_resnet50_fpn(pretrained=pre_trained, max_size=MAX_SIZE
                                         , rpn_pre_nms_top_n_train=12000
                                         , rpn_pre_nms_top_n_test=6000, rpn_post_nms_top_n_train=2000
                                         , rpn_post_nms_top_n_test=300, rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3
                                         , rpn_positive_fraction=0.7, bbox_reg_weights=(1.0, 1.0, 1.0, 1.0)
                                         , box_batch_size_per_image=32)
        # dl_model = maskrcnn_resnet50_fpn(pretrained=pre_trained, max_size=MAX_SIZE)

        # del dl_model.state_dict()["roi_heads.box_predictor.bbox_pred.weight"]
        # del dl_model.state_dict()["roi_heads.box_predictor.cls_score.weight"]
        # del dl_model.state_dict()["roi_heads.box_predictor.cls_score.bias"]
        # del dl_model.state_dict()["roi_heads.box_predictor.bbox_pred.bias"]

        # Remove incompatible parameters
        # newdict = removekey(dl_model.state_dict(), ['roi_heads.box_predictor.cls_score.bias'
        #                                             , 'roi_heads.box_predictor.cls_score.weight'
        #                                             , 'roi_heads.box_predictor.bbox_pred.bias'
        #                                             , 'roi_heads.box_predictor.bbox_pred.weight'])
        # dl_model.state_dict = newdict
        # dl_model.load_state_dict(newdict)
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
        hidden_layer = 256
        # and replace the mask predictor with a new one
        dl_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                              hidden_layer,
                                                              num_classes)
    else:
        dl_model = maskrcnn_resnet50_fpn(num_classes=numclasses, pretrained_backbone=pretrained_backbone
                                         , max_size=MAX_SIZE, rpn_head=rpnhead
                                         , rpn_anchor_generator=anchor_generator, rpn_pre_nms_top_n_train=12000
                                         , rpn_pre_nms_top_n_test=6000, rpn_post_nms_top_n_train=2000
                                         , rpn_post_nms_top_n_test=300, rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3
                                         , rpn_positive_fraction=0.7, bbox_reg_weights=(1.0, 1.0, 1.0, 1.0)
                                         , box_batch_size_per_image=32)
    return dl_model

# T.ClipBlackBorder(BG_INTENSITY),


def main():
    logging.basicConfig(filename='logs' + os.sep + 'example.log', level=logging.DEBUG)
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

    logging.info('Loading data sets')
    image_datasets = {x: DeepLesion(DIR_IN + os.sep + x, GT_FN_DICT[x], data_transforms[x]) for x in ['train', 'val'
                                                                                                      , 'test']}
    logging.info('data sets loaded')
    logging.info('Loading data loaders')
    dl_dataloaders = {x: DataLoader(image_datasets[x], batch_size=3, shuffle=True, num_workers=0
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
    #             msk_idx = np.where(mask == 1)
    #             img_copy[msk_idx[0], msk_idx[1], 0] = 255
    #         cv2.imshow(str(batch_id) + " " + str(i), img_copy)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dl_model = get_model(False, True, 2)

    params = [p for p in dl_model.parameters() if p.requires_grad]

    # Observe that not all parameters are being optimized
    optimizer_ft = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
    # optimizer_ft = Adam(params, lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=100)

    num_epochs = 10
    since = time.time()
    # best_model_wts = copy.deepcopy(dl_model.state_dict())
    # best_llf = 0
    # best_nlf = 999

    logging.info('momentum:' + str(optimizer_ft.state_dict()['param_groups'][0]['momentum']))
    logging.info('weight_decay:' + str(optimizer_ft.state_dict()['param_groups'][0]['weight_decay']))
    # logging.info('LR decay gamma:' + str(exp_lr_scheduler.state_dict()['gamma']))
    # logging.info('LR decay step size:' + str(exp_lr_scheduler.state_dict()['step_size']) + '\n')

    for epoch in range(num_epochs):
        # deep_copy_flag = False
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 20)
        train_one_epoc(dl_model, optimizer_ft, dl_dataloaders['train'], dl_dataset_sizes['train'])

        llf, nlf = evaluate(dl_model, dl_dataloaders['val'])

        logging.info('LLF: {}'.format(llf))
        logging.info('NLF: {}'.format(nlf) + '\n')

        # exp_lr_scheduler.step()

        # if llf > best_llf:
        #     deep_copy_flag = True
        #     best_nlf = nlf
        #     best_llf = llf
        # elif (llf == best_llf) & (nlf < best_nlf):
        #     deep_copy_flag = True
        #     best_nlf = nlf
        # if deep_copy_flag:
        best_model_wts = copy.deepcopy(dl_model.state_dict())
        torch.save(best_model_wts, 'saved_models' + os.sep + str(epoch) + '_deeplesion.pth')
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    main()
