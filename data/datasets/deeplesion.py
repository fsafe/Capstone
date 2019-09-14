from torch.utils.data import Dataset
import torch
from utils import *


class DeepLesion(Dataset):

    def __init__(self, data_dir, annotations_fn, transform=None):
        self.annotations_fn = annotations_fn
        self.data_dir = data_dir
        self.annotation_info = read_DL_info(annotations_fn)
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.data_dir, list(self.annotation_info.keys())[index]), -1)
        bboxes = list(self.annotation_info.values())[index][1]
        points = list(self.annotation_info.values())[index][0]
        spacing = list(self.annotation_info.values())[index][4][0]
        pseudo_masks = CreatePseudoMask(image, bboxes, points)
        targets = {"boxes": bboxes, "masks": pseudo_masks}
        if self.transform:
            image, _, targets = self.transform(image, spacing, targets)
        label = torch.ones(len(bboxes), dtype=torch.int64)
        targets["labels"] = label
        return image, targets

    def __len__(self):
        return len(self.annotation_info)
