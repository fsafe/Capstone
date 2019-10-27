import sys
import math
import torch
from utils import getioulist, calc_froc_metrics


def train_one_epoc(model, optimizer, scheduler, dataloader, dataset_size):

    running_loss = 0.0
    model.train()  # Set model to training mode

    # Iterate over data.
    for inputs, targets in dataloader:

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history for training
        with torch.set_grad_enabled(True):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        losses.backward()
        optimizer.step()
        # scheduler.step()

        # statistics
        running_loss += losses.item() * len(inputs)

    epoch_loss = running_loss / dataset_size
    print('Train Loss: {:.4f}'.format(epoch_loss))


def evaluate(model, dataloader):
    model.eval()  # Set model to training mode
    ioulist = []
    for inputs, targets in dataloader:
        with torch.set_grad_enabled(False):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            output = model(inputs)
        ioulist += getioulist(output, targets)
    return calc_froc_metrics(ioulist)

# def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     # lesion_localization = 0
#     # non_lesion_localization = 0
#     # best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train']:
#             if phase == 'train':
#                 running_loss = 0.0
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode
#
#             # running_corrects = 0
#
#             # Iterate over data.
#             for inputs, targets in dataloaders[phase]:
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#                     model = model.to(device)
#                     if phase == 'train':
#                         loss_dict = model(inputs, targets)
#                         losses = sum(loss for loss in loss_dict.values())
#                     else:
#                         output = model(inputs)
#                         ioulist = getioulist(output, targets)
#                         forc_stats = calc_froc_metrics(ioulist)
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         losses.backward()
#                         optimizer.step()
#                         # scheduler.step()
#
#                 # statistics
#                 if phase == 'train':
#                     running_loss += losses.item() * len(inputs)
#                     # running_corrects += torch.sum(preds == targets["labels"])
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             # epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#             #     phase, epoch_loss, epoch_acc))
#             print('{} Loss: {:.4f}'.format(phase, epoch_loss))
#
#             # deep copy the model
#             # if phase == 'val' and epoch_acc > best_acc:
#             #     best_acc = epoch_acc
#             #     best_model_wts = copy.deepcopy(model.state_dict())
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     # print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     # model.load_state_dict(best_model_wts)
#     # # Export the model
#     # torch.onnx.export(model,  # model being run
#     #                   (inputs, targets),  # model input (or a tuple for multiple inputs)
#     #                   "deep_lesion.onnx",  # where to save the model (can be a file or file-like object)
#     #                   export_params=True,  # store the trained parameter weights inside the model file
#     #                   opset_version=10,  # the ONNX version to export the model to
#     #                   do_constant_folding=True,  # wether to execute constant folding for optimization
#     #                   input_names=['input'],  # the model's input names
#     #                   output_names=['output'],  # the model's output names
#     #                   dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
#     #                                 'output': {0: 'batch_size'}})
