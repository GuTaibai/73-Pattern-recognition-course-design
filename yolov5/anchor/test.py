# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/24 20:23
@File    : test.py
@Author  : zj
@Description: 
"""

import numpy as np

from gen_anchors import Model, Dataset, get_yolov5_data
from autoanchor import check_anchor_order, check_anchors


def test_dataset():
    shapes, labels = get_yolov5_data("../datasets/", "train4")
    dataset = Dataset(shapes, labels)

    anchors = np.array([
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ])
    stride = [8, 16, 32]
    m = Model(anchors=anchors, stride=stride)

    check_anchors(dataset, m)
    # print('anchors:', m.anchors)

    return m


def test_model(m=None):
    if m is None:
        anchors = np.array([
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ])
        stride = [8, 16, 32]
        m = Model(anchors=anchors, stride=stride)

    print("anchors:", m.anchors)
    print("stride:", m.stride)
    check_anchor_order(m)
    print("anchors:", m.anchors)
    print("stride:", m.stride)


if __name__ == '__main__':
    m = test_dataset()
    test_model(m)
