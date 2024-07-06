# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 20:01
@File    : gen_anchors.py
@Author  : zj
@Description:

Usage - Generate anchors for VOC:
    $ python3 v5/gen_anchors.py ../datasets/ ./output/
    $ python3 v5/gen_anchors.py ../datasets/ ./output/ -n 5

Usage - Generate anchors for COCO:
    $ python3 v5/gen_anchors.py ../datasets/coco/ -t train2017 -v val2017 ./output/ -e coco
    $ python3 v5/gen_anchors.py ../datasets/coco/ -t train2017 -v val2017 ./output/ -e coco -n 5

"""

import os
import cv2
import glob
import torch
import argparse

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from autoanchor import kmean_anchors


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Anchor-boxes.")
    parser.add_argument('data', metavar='DIR', help='Path to dataset')
    parser.add_argument('-t', '--train', metavar='TRAIN', default='voc2yolov5-train', help='Train dataset')
    parser.add_argument('-v', '--val', metavar='VAL', default='voc2yolov5-val', help='Val dataset')
    parser.add_argument('output', metavar='OUTPUT', help='Path to save files')
    parser.add_argument('-e', '--exp', metavar='EXP', default='voc', help='Sub-folder name')

    parser.add_argument('-n', '--num-clusters', metavar='NUM', default=None,
                        help='Number of cluster centroids')

    args = parser.parse_args()
    print("args:", args)

    return args


class Model:

    def __init__(self, anchors=None, stride=None):
        """

        :param anchors: [Nl, Na, 2], where Nl is the number of detection layers, Na is the number of anchors, and 2 is the width and height of each anchor
        :param stride: [Nl], where Nl is the scaling factor of each detection layer relative to the input image
        """
        self.anchors = torch.from_numpy(np.array(anchors, dtype=float)).view(len(anchors), -1, 2)
        self.stride = torch.from_numpy(np.array(stride, dtype=int))
        assert len(self.anchors) == len(self.stride)


class Dataset:

    def __init__(self, shapes=None, labels=None):
        """
        :param shapes: [N, 2], where N is the number of images and 2 is the width and height of each image
        :param labels: [N, ...], where N is the number of images, each item in the list contains the annotation box width and height of the corresponding image
        """
        self.shapes = shapes
        self.labels = labels
        assert len(self.shapes) == len(self.labels)


def get_yolov5_data(root, name):
    image_dir = os.path.join(root, name, "images")
    assert os.path.isdir(image_dir), image_dir
    label_dir = os.path.join(root, name, 'labels')
    assert os.path.isdir(label_dir), label_dir
    label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

    shapes = list()
    labels = list()

    for label_path in tqdm(label_path_list):
        # [[box_w, box_h], ]
        # The coordinate size is relative to the width and height of the image
        boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
        if len(boxes) == 0:
            continue
        if len(boxes.shape) == 1:
            boxes = [boxes]
        # for label, xc, yc, box_w, box_h in boxes:
        #     box_list.append([box_w, box_h])
        labels.append(np.array(boxes))

        image_name = os.path.basename(label_path).replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        assert os.path.isfile(image_path), image_path
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        shapes.append((w, h))

    return np.array(shapes), labels


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids: ndarray, train_dataset: Dataset, val_dataset: Dataset, anchor_file: str,
                          img_size: int = 640):
    # Get label wh
    train_shapes = img_size * train_dataset.shapes / train_dataset.shapes.max(1, keepdims=True)
    train_wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(train_shapes, train_dataset.labels)])  # wh
    train_avg_iou = avg_IOU(train_wh0, centroids)

    val_shapes = img_size * val_dataset.shapes / val_dataset.shapes.max(1, keepdims=True)
    val_wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(val_shapes, val_dataset.labels)])  # wh
    test_avg_iou = avg_IOU(val_wh0, centroids)

    print(f'Train Avg IOU: {train_avg_iou}')
    print(f'Test Avg IOU: {test_avg_iou}')

    print(f"Write to {anchor_file}")
    with open(anchor_file, 'w') as f:
        anchors = [str('%.2f' % x) for x in centroids.reshape(-1)]
        scaled_anchors = [str('%.2f' % (x / img_size)) for x in centroids.reshape(-1)]

        f.write(','.join(anchors) + '\n')
        f.write(','.join(scaled_anchors) + '\n')

        f.write('%f\n' % (train_avg_iou))
        f.write('%f\n' % (test_avg_iou))
        print()


def main():
    args = parse_args()

    train_shapes, train_labels = get_yolov5_data(args.data, args.train)
    train_dataset = Dataset(train_shapes, train_labels)
    val_shapes, val_labels = get_yolov5_data(args.data, args.val)
    val_dataset = Dataset(val_shapes, val_labels)

    output_dir = os.path.join(args.output, args.exp)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if args.num_clusters is None:
        for ni in range(1, 11):
            num_clusters = ni
            anchors = kmean_anchors(train_dataset, n=num_clusters, img_size=640, thr=4.0, gen=1000, verbose=False)
            write_anchors_to_file(anchors, train_dataset, val_dataset,
                                  os.path.join(output_dir, f'anchors{num_clusters}.txt'))
    else:
        num_clusters = int(args.num_clusters)
        anchors = kmean_anchors(train_dataset, n=num_clusters, img_size=640, thr=4.0, gen=1000, verbose=False)
        write_anchors_to_file(anchors, train_dataset, val_dataset,
                              os.path.join(output_dir, f'anchors{num_clusters}.txt'))


if __name__ == '__main__':
    main()
