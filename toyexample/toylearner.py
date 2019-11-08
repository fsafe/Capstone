import cv2
import time
import gc
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torch.optim import lr_scheduler, SGD
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# load image from disk
img = cv2.imread('004408_01_02_088.png', -1)

# subtract 32768from the pixel intensity to obtain the original Hounsfield unit (HU) values
# https://nihcc.app.box.com/v/DeepLesion/file/306056134060
img = img.astype(np.float32, copy=False) - 32768

# intensity windowing with window [-1024, 3071] HU covers the intensity ranges of the lung, soft tissue, and bone.
# (https://arxiv.org/pdf/1806.09648.pdf)
# convert the intensities in a certain range (“window”) to 0-255 for viewing.
img -= -1024
img /= 3071 + 1024
img[img > 1] = 1
img[img < 0] = 0
img *= 255
img = img.astype('uint8')

# convert image to tensor. The output tensor will have range [0,1]
img_T = TF.to_tensor(img)

# create numpy array version of img_T Tensor and adding a blue pseudo_mask
# the addition of the mask and bounding box does not affect original image tensor (img_T)
# on this numpy version by combining 4 quarter sized ellipses. Also add a green bounding box.
# https://arxiv.org/pdf/1901.06359.pdf
img_copy = [img_T.squeeze().numpy()] * 3
img_copy = cv2.merge(img_copy)
bbox = np.array([188.354, 159.003, 223.22, 183.271])
bbox = np.int16(bbox)
cen = np.array([212.17824058, 171.81745919])
semi_axes = np.array([7, 7, 17, 6])
angles = np.array([0.94002174, -179.05997826])
cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple(semi_axes[0:2]), angles[0], 0, 90, 255, -1)
cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple(semi_axes[2:0:-1]), angles[1], -90, 0, 255, -1)
cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple(semi_axes[2:4]), angles[1], 0, 90, 255, -1)
cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple([semi_axes[0], semi_axes[3]]), angles[0], -90, 0, 255, -1)
cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

# extract pseudo_mask by identifying pixels which are colored blue
pseudo_mask = np.logical_and(img_copy[:, :, 0] == 255, img_copy[:, :, 1] == 0, img_copy[:, :, 2] == 0).astype('uint8')
pseudo_mask_T = torch.from_numpy(pseudo_mask)

# construct inputs to model
inputs = [img_T]
bbox_T = torch.from_numpy(bbox).float()
bboxes = [bbox_T]
bboxes = torch.stack(bboxes)
masks = [pseudo_mask_T]
masks = torch.stack(masks)
label = torch.ones(len(bboxes), dtype=torch.int64)
elem = {'boxes': bboxes, 'masks': masks, 'labels': label}
targets = [elem]

# # uncomment this block to check if inputs to model can be displayed correctly
# for (image, target) in zip(inputs, targets):
#     img_display = image.squeeze().numpy()
#     images_disp = [img_display] * 3
#     images_disp = [im.astype(float) for im in images_disp]
#     img_display = cv2.merge(images_disp)
#     for (bbox_disp, pseudo_mask_disp) in zip(target["boxes"], target["masks"]):
#         bbox_disp = bbox_disp.squeeze().numpy()
#         bbox_disp = np.int16(bbox)
#         mask_disp = pseudo_mask_disp.squeeze().numpy()
#         cv2.rectangle(img_display, (bbox_disp[0], bbox_disp[1]), (bbox_disp[2], bbox_disp[3]), (0, 255, 0), 1)
#         msk_idx = np.where(mask_disp == 1)
#         img_display[msk_idx[0], msk_idx[1], 0] = 255
#     cv2.imshow('original', img_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# crop image by clipping black borders
img_crop = img_T.squeeze().numpy()
u, d, l, r = (115, 430, 0, 511)
img_crop = img_crop[u:d + 1, l:r + 1]
bbox_crop = np.array([0, 0, 0, 0])
bbox_crop[0] = bbox[0] - l
bbox_crop[1] = bbox[1] - u
bbox_crop[2] = bbox[2] - l
bbox_crop[3] = bbox[3] - u
bbox_crop = np.int16(bbox_crop)
pseudo_mask_crop = pseudo_mask[u:d + 1, l:r + 1]

# construct inputs to model
img_crop_T = TF.to_tensor(img_crop)
inputs_crop = [img_crop_T]
bbox_crop_T = torch.from_numpy(bbox_crop).float()
bboxes_crop = [bbox_crop_T]
bboxes_crop = torch.stack(bboxes_crop)
pseudo_mask_crop_T = torch.from_numpy(pseudo_mask_crop)
masks_crop = [pseudo_mask_crop_T]
masks_crop = torch.stack(masks_crop)
label_crop = torch.ones(len(bboxes_crop), dtype=torch.int64)
elem_crop = {'boxes': bboxes_crop, 'masks': masks_crop, 'labels': label_crop}
targets_crop = [elem_crop]

# # uncomment this block to check if inputs to model can be displayed correctly
# for (image, target) in zip(inputs_crop, targets_crop):
#     img_display = image.squeeze().numpy()
#     images_disp = [img_display] * 3
#     images_disp = [im.astype(float) for im in images_disp]
#     img_display = cv2.merge(images_disp)
#     for (bbox_disp, pseudo_mask_disp) in zip(target["boxes"], target["masks"]):
#         bbox_disp = bbox_disp.squeeze().numpy()
#         bbox_disp = np.int16(bbox_disp)
#         mask_disp = pseudo_mask_disp.squeeze().numpy()
#         cv2.rectangle(img_display, (bbox_disp[0], bbox_disp[1]), (bbox_disp[2], bbox_disp[3]), (0, 255, 0), 1)
#         msk_idx = np.where(mask_disp == 1)
#         img_display[msk_idx[0], msk_idx[1], 0] = 255
#     cv2.imshow('cropped', img_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

pretrained = True

if pretrained:
    model = maskrcnn_resnet50_fpn(pretrained=True, min_size=512, max_size=512)
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 2  # 1 class (lesion) + 0 (background)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    params = [p for p in model.parameters() if p.requires_grad]

    # Observe that not all parameters are being optimized
    optimizer_ft = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)

else:
    # Observe that all parameters are being optimized
    model = maskrcnn_resnet50_fpn(num_classes=2)
    optimizer_ft = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# don't know how to initialize the weights of the model
# torch.nn.init.kaiming_normal_(model.parameters(), mode='fan_out')

# Decay LR by a factor of 0.1 every 2 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)

num_epochs = 100
since = time.time()
model.train()

print('Pretrained:' + str(pretrained))
print('momentum:' + str(optimizer_ft.state_dict()['param_groups'][0]['momentum']))
print('weight_decay:' + str(optimizer_ft.state_dict()['param_groups'][0]['weight_decay']))
print('LR decay gamma:' + str(exp_lr_scheduler.state_dict()['gamma']))
print('LR decay step size:' + str(exp_lr_scheduler.state_dict()['step_size']))

for epoch in range(num_epochs):
    print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    print('lr:' + str(optimizer_ft.state_dict()['param_groups'][0]['lr']))
    # valid inputs to model are (inputs, targets) or (inputs_crop, targets_crop)
    loss_dict = model(inputs, targets)
    for (k, i) in loss_dict.items():
        print(str(k) + ':' + str(i))
    losses = sum(loss for loss in loss_dict.values())
    del loss_dict
    print('Total Train Loss: {:.4f}'.format(losses.item()))

    # zero the parameter gradients
    optimizer_ft.zero_grad()
    # perform backward propagation, optimization and update model parameters
    losses.backward()
    optimizer_ft.step()
    exp_lr_scheduler.step()
    del losses
    gc.collect()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
