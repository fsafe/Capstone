import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch.utils.data import DataLoader
from data.datasets.deeplesion import *
from data import transforms as T
from data.collate_batch import BatchCollator

# A single windowing (−1024 to 3071 HU) that covers the
# intensity ranges of lung, soft tissue, and bone.
# For more information on Intensity Windowing go to
# https://radiopaedia.org/articles/windowing-ct
WINDOWING = [-1024, 3071]
BG_INTENSITY = 32000
INTENSITY_OFFSET = 32768
NORM_SPACING = 0.8  # Resize every image slice so that each pixel corresponds to 0.8mm
MAX_SIZE = 512

DIR_IN = 'Images_png'  # input directory
GT_FN_TRAIN = 'DL_info_train.csv'  # Ground truth file for training data
GT_FN_VAL = 'DL_info_val.csv'  # Ground truth file for validation data
GT_FN_TEST = 'DL_info_test_sample.csv'  # Ground truth file for test data
GT_FN_DICT = {"train": GT_FN_TRAIN, "val": GT_FN_VAL, "test": GT_FN_TEST}


@torch.no_grad()
def test_model(model, dataloader):
    model.eval()  # Set model to training mode
    for inputs, targets in dataloader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        output = model(inputs)
    return output


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
    model.load_state_dict(torch.load('saved_models\\0_deeplesion.pth', map_location='cpu'))
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
                            # , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
    }
    image_datasets = {x: DeepLesion(DIR_IN + os.sep + x, GT_FN_DICT[x], data_transforms[x]) for x in ['train', 'val'
                                                                                                      , 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=3, shuffle=True, num_workers=0
                                 , collate_fn=BatchCollator) for x in ['train', 'val', 'test']}
    output = test_model(model, dataloaders['test'])
    predbox = output[0]['boxes'][4].numpy()
    # predmask = output[0]['masks'][4].squeeze().numpy()
    # print(output[0]['scores'][4].numpy())
    for batch_id, (inputs, targets) in enumerate(dataloaders['test']):
        i = 0
        for i, (image, target) in enumerate(zip(inputs, targets)):
            img_copy = image.squeeze().numpy()
            images = [img_copy] * 3
            images = [im.astype(float) for im in images]
            img_copy = cv2.merge(images)
            for j, (bbox, pseudo_mask) in enumerate(zip(target["boxes"], target["masks"])):
                bbox = target["boxes"][j].squeeze().numpy()
                bbox = np.int16(bbox)
                mask = target["masks"][j].squeeze().numpy()
                cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                cv2.rectangle(img_copy, (predbox[0], predbox[1]), (predbox[2], predbox[3]), (0, 0, 255), 1)
                msk_idx = np.where(mask == 1)
                # pmsk_idx = np.where(predmask == 1)
                img_copy[msk_idx[0], msk_idx[1], 0] = 255
                # img_copy[pmsk_idx[0], pmsk_idx[1], 0] = 255
            # cv2.imshow(str(batch_id) + " " + str(i), img_copy)
            cv2.imwrite('simple_test\\test_sample.jpg', img_copy*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()