from torch.utils.data import Dataset
from utils import *


class DeepLesion(Dataset):

    def __init__(self, data_dir, annotations_fn, transform=None):
        self.annotations_fn = annotations_fn
        self.data_dir = data_dir
        self.annotation_info = read_DL_info(annotations_fn)
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        fname = list(self.annotation_info.keys())[index]
        image = cv2.imread(os.path.join(self.data_dir, fname), -1)
        bboxes = list(self.annotation_info.values())[index][1]
        points = list(self.annotation_info.values())[index][0]
        spacing = list(self.annotation_info.values())[index][4][0]
        pseudo_masks = CreatePseudoMask(image, bboxes, points)
        targets = {"boxes": bboxes, "masks": pseudo_masks}
        if self.transform:
            image, _, targets = self.transform(image, spacing, targets)
            record = self.annotation_info.get(fname)
            tsize = np.array([image.shape[1], image.shape[2]])
            if len(record) > 8:
                record[8] = tsize
            else:
                record.append(tsize)
        label = torch.ones(len(bboxes), dtype=torch.int64).to(self.device)
        targets["labels"] = label
        targets["image_id"] = fname
        return image, targets

    def __len__(self):
        return len(self.annotation_info)
