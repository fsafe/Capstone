import numpy as np
from collections import OrderedDict
import csv
import cv2
import sys
import os
import traceback
from scipy.ndimage.morphology import binary_opening
from scipy.spatial import distance
import torch


# https://github.com/rsummers11/CADLab/blob/1192f13b1a6fc0beb3407534a9d3ef7b59df6ba0/lesion_detector_3DCE/rcnn/fio/load_ct_img.py
def get_mask(im, th=32000):
    # use a intensity threshold to roughly find the mask of the body
    # 32000 is the approximate background intensity value
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
    # mask = binary_dilation(mask)
    # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

    if mask.sum() == 0:  # maybe atypical intensity
        mask = im * 0 + 1
    return mask.astype(dtype=np.int32)


# https://github.com/rsummers11/CADLab/blob/1192f13b1a6fc0beb3407534a9d3ef7b59df6ba0/lesion_detector_3DCE/rcnn/fio/load_ct_img.py
def windowing(im, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


# https://github.com/rsummers11/CADLab/blob/1192f13b1a6fc0beb3407534a9d3ef7b59df6ba0/lesion_detector_3DCE/rcnn/fio/load_ct_img.py
def get_range(mask, margin=0):
    idx = np.nonzero(mask)
    up = max(0, idx[0].min() - margin)
    down = min(mask.shape[0] - 1, idx[0].max() + margin)
    left = max(0, idx[1].min() - margin)
    right = min(mask.shape[1] - 1, idx[1].max() + margin)
    return up, down, left, right


# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/3252222#3252222
def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/3252222#3252222
def seq_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def ellipse_quadrant_params(a1, a2, b1, b2):
    cen = seq_intersect(a1, a2, b1, b2)
    right_major, left_major = (a1, a2) if a1[0] >= a2[0] else (a2, a1)
    upper_minor, lower_minor = (b1, b2) if b1[1] >= b2[1] else (b2, b1)

    d1 = distance.euclidean(right_major, cen)
    d2 = distance.euclidean(upper_minor, cen)
    d3 = distance.euclidean(left_major, cen)
    d4 = distance.euclidean(lower_minor, cen)

    dif_1_4 = right_major - cen
    dif_2_3 = left_major - cen

    ang_1_4 = (np.arctan2(dif_1_4[1], dif_1_4[0]) * 180 / np.pi)
    ang_2_3 = (np.arctan2(dif_2_3[1], dif_2_3[0]) * 180 / np.pi)

    return cen, np.array([d1, d2, d3, d4]), np.array([ang_1_4, ang_2_3])


def read_DL_info(fname):
    annotation_info = OrderedDict()
    try:
        with open(fname, 'rt') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip the headers row
            for row in reader:
                if row[10] == '0':
                    filename = row[0]  # replace the last _ in filename with / or \
                    idx = filename.rindex('_')
                    row[0] = filename[:idx] + os.sep + filename[idx + 1:]
                    record = annotation_info.get(row[0])
                    if record is not None:
                        recist_points_list = annotation_info.get(row[0])[0]
                        bboxes_list = annotation_info.get(row[0])[1]
                        noisy_list = annotation_info.get(row[0])[2]
                        spacing3_d_list = annotation_info.get(row[0])[3]
                        spacing_list = annotation_info.get(row[0])[4]
                        size2_d_list = annotation_info.get(row[0])[5]
                        size_list = annotation_info.get(row[0])[6]
                        train_val_test_list = annotation_info.get(row[0])[7]
                    else:
                        recist_points_list = []
                        bboxes_list = []
                        noisy_list = []
                        spacing3_d_list = []
                        spacing_list = []
                        size2_d_list = []
                        size_list = []
                        train_val_test_list = []
                        if annotation_info.get(row[0]) is None:
                            annotation_info[row[0]] = []
                        annotation_info[row[0]].append(recist_points_list)
                        annotation_info[row[0]].append(bboxes_list)
                        annotation_info[row[0]].append(noisy_list)
                        annotation_info[row[0]].append(spacing3_d_list)
                        annotation_info[row[0]].append(spacing_list)
                        annotation_info[row[0]].append(size2_d_list)
                        annotation_info[row[0]].append(size_list)
                        annotation_info[row[0]].append(train_val_test_list)
                    recist_points_list.append(np.array([float(x) for x in row[5].split(',')]))
                    bboxes_list.append(np.array([float(x) - 1 for x in row[6].split(',')]))
                    noisy_list.append(int(row[10]) > 0)
                    sp3_d_value = np.array([float(x) for x in row[12].split(',')])
                    spacing3_d_list.append(sp3_d_value)
                    spacing_list.append(sp3_d_value[0])
                    size2_d_value = np.array([float(x) for x in row[13].split(',')])
                    size2_d_list.append(size2_d_value)
                    size_list.append(size2_d_value[0])
                    train_val_test_list.append(int(row[17]))
    except FileNotFoundError:
        print(traceback.format_exc())
        sys.exit(1)
    return annotation_info


def CreatePseudoMask(image, bboxes, diagonal_points_list):
    img_copy = image.copy()
    images = [img_copy] * 3
    images = [im.astype(float) for im in images]
    pseudo_masks = []
    for bbox, diagonal_points in zip(bboxes, diagonal_points_list):
        img_copy = cv2.merge(images)
        a1 = np.array([diagonal_points[0], diagonal_points[1]])
        a2 = np.array([diagonal_points[2], diagonal_points[3]])
        b1 = np.array([diagonal_points[4], diagonal_points[5]])
        b2 = np.array([diagonal_points[6], diagonal_points[7]])
        cen, semi_axes, angles = ellipse_quadrant_params(a1, a2, b1, b2)
        semi_axes = semi_axes.astype(int)
        if type(bbox) == torch.Tensor:
            bbox_copy = bbox.numpy()
        else:
            bbox_copy = bbox.copy()
        bbox_copy = np.int16(bbox_copy)
        cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple(semi_axes[0:2]), angles[0], 0, 90, 255, -1)
        cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple(semi_axes[2:0:-1]), angles[1], -90, 0, 255, -1)
        cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple(semi_axes[2:4]), angles[1], 0, 90, 255, -1)
        cv2.ellipse(img_copy, tuple(cen.astype(int)), tuple([semi_axes[0], semi_axes[3]]), angles[0], -90, 0, 255, -1)
        cv2.rectangle(img_copy, (bbox_copy[0], bbox_copy[1]), (bbox_copy[2], bbox_copy[3]), (0, 255, 0), 1)
        pseudo_masks.append(np.logical_and(img_copy[:, :, 0] == 255, img_copy[:, :, 1] == 0, img_copy[:, :, 2] == 0))
    return pseudo_masks


def getioulist(output_list, targets_list):
    ioulist = []
    for (item_gtlabels, item_predlabels) in zip(targets_list, output_list):
        iou_pred_array = np.full((item_predlabels['boxes'].shape[0], item_gtlabels['boxes'].shape[0] + 1), -1.00)
        for pindex, (pred_bbox, pred_scores) in enumerate(zip(item_predlabels['boxes'], item_predlabels['scores'])):
            for gtindex, gt_bbox in enumerate(item_gtlabels['boxes']):
                iou_pred_array[pindex, gtindex] = calculate_iou(gt_bbox, pred_bbox)
            iou_pred_array[pindex, item_gtlabels['boxes'].shape[0]] = pred_scores
        ioulist.append(iou_pred_array)
    return ioulist


def calculate_iou(bb1, bb2):
    dx = min(max(bb1[0], bb1[2]), max(bb2[0], bb2[2])) - max(min(bb1[0], bb1[2]), min(bb2[0], bb2[2]))
    dy = min(max(bb1[1], bb1[3]), max(bb2[1], bb2[3])) - max(min(bb1[1], bb1[3]), min(bb2[1], bb2[3]))
    if (dx > 0) and (dy > 0):
        return 2 * dx * dy / (
                (abs(bb1[0] - bb1[2]) * abs(bb1[1] - bb1[3])) + (abs(bb2[0] - bb2[2]) * abs(bb2[1] - bb2[3])))
    else:
        return 0


def calc_froc(ious, detection_threshold=0.50, iou_threshold=0.50):
    num_images = 0
    lesion_localizations = 0
    non_lesion_localizations = 0
    num_lesions = 0
    for i, elem in enumerate(ious):
        num_lesions += elem.shape[1]-1
        pos_detections = elem[np.where(elem[:, -1] >= detection_threshold)]
        lesion_localizations += (np.amax(pos_detections[:, :-1], axis=1) >= iou_threshold).sum()
        non_lesion_localizations += (np.amax(pos_detections[:, :-1], axis=1) < iou_threshold).sum()
        num_images += 1
    llf = lesion_localizations/num_lesions
    nlf = non_lesion_localizations/num_images
    return llf, nlf
