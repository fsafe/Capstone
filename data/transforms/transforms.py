from utils import *
from torchvision.transforms import functional as TF
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, spacing=None, targets=None):
        for t in self.transforms:
            image, spacing, targets = t(image, spacing, targets)
        return image, spacing, targets


class ToOriginalHU(object):
    """Subtracting offset from the16-bit pixel intensities to
    obtain the original Hounsfield Unit (HU) values"""

    def __init__(self, offset):
        self.offset = offset

    def __call__(self, image, spacing=None, targets=None, ):
        image = image.astype(np.float32, copy=False) - self.offset
        return image, spacing, targets


class IntensityWindowing(object):
    """Scale intensity from win[0]~win[1] to float numbers in 0~255
       and convert the windowed image to an 8-bit image."""

    def __init__(self, win):
        self.win = win

    def __call__(self, image, spacing=None, targets=None):
        # scale intensity to float numbers in 0~255
        windowing(image, self.win)
        image = windowing(image, self.win).astype('uint8')
        return image, spacing, targets


class ClipBlackBorder(object):
    """Clip black borders in image for computational efficiency
       and adjust bounding box accordingly"""

    def __init__(self, int_th):
        self.intensity_threshold = int_th

    def __call__(self, image, spacing=None, targets=None):
        mask = get_mask(image, self.intensity_threshold)
        u, d, l, r = get_range(mask)
        image = image[u:d + 1, l:r + 1]
        if targets is not None:
            for i, (bbox, pseudo_mask) in enumerate(zip(targets["boxes"], targets["masks"])):
                pseudo_mask = pseudo_mask[u:d + 1, l:r + 1]
                bbox[0] -= l
                bbox[1] -= u
                bbox[2] -= l
                bbox[3] -= u
                targets["boxes"][i] = bbox
                targets["masks"][i] = pseudo_mask
        return image, spacing, targets


class SpacingResize(object):
    """resize according to spacing"""

    def __init__(self, norm_spacing, max_size):
        self.norm_spacing = norm_spacing
        self.max_size = max_size

    def __call__(self, image, spacing, targets=None):
        im_shape = image.shape[0:2]
        im_scale = float(spacing) / self.norm_spacing
        max_shape = np.max(im_shape) * im_scale
        if max_shape > self.max_size:
            im_scale1 = float(self.max_size) / max_shape
            im_scale *= im_scale1
        image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_AREA)
        if targets is not None:
            for i, (bbox, pseudo_mask) in enumerate(zip(targets["boxes"], targets["masks"])):
                pseudo_mask = pseudo_mask.astype(np.float)
                pseudo_mask = cv2.resize(pseudo_mask, None, None, fx=im_scale, fy=im_scale
                                         , interpolation=cv2.INTER_NEAREST)
                pseudo_mask = pseudo_mask.astype(np.bool)
                bbox *= im_scale
                targets["boxes"][i] = bbox
                targets["masks"][i] = pseudo_mask
        return image, spacing, targets


class ToTensor(object):
    def __call__(self, image, spacing=None, targets=None):
        image = TF.to_tensor(image)
        if targets is not None:
            for i, (bbox, pseudo_mask) in enumerate(zip(targets["boxes"], targets["masks"])):
                pseudo_mask = TF.to_tensor(pseudo_mask)
                if type(bbox) != torch.Tensor:
                    bbox = torch.from_numpy(bbox).float()
                else:
                    bbox = bbox.float()
                targets["boxes"][i] = bbox
                targets["masks"][i] = pseudo_mask
            targets['boxes'] = torch.stack(targets['boxes'])
            targets['masks'] = torch.stack(targets['masks'])
        return image, spacing, targets
