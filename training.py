import time
import copy
from torch.optim import lr_scheduler, Adam
from data import transforms as T
from data.collate_batch import BatchCollator
from torch.utils.data import DataLoader
from data.datasets.deeplesion import *
from torchvision.models.detection import maskrcnn_resnet50_fpn

# A single windowing (−1024 to 3071 HU) that covers the
# intensity ranges of lung, soft tissue, and bone.
# For more information on Intensity Windowing go to
# https://radiopaedia.org/articles/windowing-ct
WINDOWING = [-1024, 3071]
BG_INTENSITY = 32000
INTENSITY_OFFSET = 32768
NORM_SPACING = 0.8  # Resize every image slice so that each pixel corresponds to 0.8mm
MAX_SIZE = 512
GT_FN = 'DL_info.csv'  # Ground truth file
DIR_IN = 'Images_png'  # input directory


def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                # inputs.to(device)
                # targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss_dict = model(inputs, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        losses.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += losses.item() * len(inputs)
                # running_corrects += torch.sum(preds == targets["labels"])

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


def main():
    data_transforms = {
        'train': T.Compose([T.ClipBlackBorder(BG_INTENSITY), T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
    }

    image_datasets = {x: DeepLesion(DIR_IN, GT_FN, data_transforms[x]) for x in ['train']}
    dl_dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=0
                                    , collate_fn=BatchCollator) for x in ['train']}
    dl_dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

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

    dl_model = maskrcnn_resnet50_fpn(num_classes=2)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(dl_model.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = Adam(dl_model.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(dl_model, optimizer_ft, exp_lr_scheduler, dl_dataloaders, dl_dataset_sizes, 1)


if __name__ == '__main__':
    main()
