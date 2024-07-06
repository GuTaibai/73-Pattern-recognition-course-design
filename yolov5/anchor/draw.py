'''
import numpy as np
import cv2

# 定义锚点
anchors_voc = [
    [0.9531, 1.1440, 1.4713, 1.4238, 3.1402, 2.2344],  # P3/8
    [4.8121, 1.9816, 8.3975, 2.8052, 21.8237, 2.2261],  # P4/16
    [18.6426, 1.4270, 19.4326, 2.3723, 18.0818, 4.5073]  # P5/32
]


def draw_anchors(anchors, canva_name="anchors"):
    canva_list = []
    final_canva_w = 0
    final_canva_h = 0

    for i, scale_anchors in enumerate(anchors):
        scale_canva_list = []
        for j in range(0, len(scale_anchors), 2):
            anchor_w = scale_anchors[j]
            anchor_h = scale_anchors[j + 1]
            canva_w = int(anchor_w + 4)
            canva_h = int(anchor_h + 4)
            canva = np.ones((canva_h, canva_w, 3), dtype=np.uint8) * 255

            xmin = (canva_w - anchor_w) / 2.
            ymin = (canva_h - anchor_h) / 2.
            xmax = (canva_w + anchor_w) / 2.
            ymax = (canva_h + anchor_h) / 2.
            cv2.rectangle(canva, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 1)

            scale_canva_list.append(canva)
            final_canva_w += canva_w
            final_canva_h = max(final_canva_h, canva_h)

        canva_list.append(scale_canva_list)

    final_canva_w += 10 * (len(canva_list[0]) + 1)
    final_canva_h += 10 * (len(canva_list) + 1)
    final_canva = np.ones((final_canva_h, final_canva_w, 3), dtype=np.uint8) * 255

    center_h = 10
    for scale_canva_list in canva_list:
        center_w = 10
        for canva in scale_canva_list:
            ymin = max(0, int(center_h))
            ymax = min(final_canva.shape[0], ymin + canva.shape[0])
            xmin = max(0, int(center_w))
            xmax = min(final_canva.shape[1], xmin + canva.shape[1])

            final_canva[ymin:ymax, xmin:xmax] = canva[:ymax - ymin, :xmax - xmin]

            center_w += canva.shape[1] + 10

        center_h += final_canva_h // len(canva_list) + 10

    cv2.imshow(canva_name, final_canva)
    cv2.waitKey(0)
    return final_canva


# 绘制锚点
final_canva = draw_anchors(anchors_voc)
cv2.imwrite("anchors_visualization.jpg", final_canva)
'''
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 给定的anchors和stride参数
anchors = np.array([[[ 0.9531,  1.1440],
                     [ 1.4713,  1.4238],
                     [ 3.1402,  2.2344]],

                    [[ 4.8121,  1.9816],
                     [ 8.3975,  2.8052],
                     [21.8237,  2.2261]],

                    [[18.6426,  1.4270],
                     [19.4326,  2.3723],
                     [18.0818,  4.5073]]])

strides = np.array([8, 16, 32])

def plot_anchors(anchors, strides, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Visualization of Anchor Boxes at Different Strides', fontsize=16)

    for i in range(len(strides)):
        stride = strides[i]
        ax = axs[i]
        ax.set_title(f'Stride: {stride}', fontsize=14)
        ax.set_xlim(0, stride * 10)
        ax.set_ylim(0, stride * 10)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.grid(True)

        for anchor in anchors[i]:
            width = anchor[0] * stride
            height = anchor[1] * stride
            rect = patches.Rectangle((stride * 5, stride * 5), width, height,
                                     linewidth=2, edgecolor='b', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.show()

# 保存并显示图像
save_path = "anchor_boxes_visualization.png"
plot_anchors(anchors, strides, save_path)
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 给定的anchors和stride参数
anchors = np.array([[[ 0.9531,  1.1440],
                     [ 1.4713,  1.4238],
                     [ 3.1402,  2.2344]],

                    [[ 4.8121,  1.9816],
                     [ 8.3975,  2.8052],
                     [21.8237,  2.2261]],

                    [[18.6426,  1.4270],
                     [19.4326,  2.3723],
                     [18.0818,  4.5073]]])

strides = np.array([8, 16, 32])

def plot_anchors(anchors, strides, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Visualization of Anchor Boxes at Different Strides', fontsize=16)

    for i in range(len(strides)):
        stride = strides[i]
        ax = axs[i]
        ax.set_title(f'Stride: {stride}', fontsize=14)
        max_size = max((anchors[i][:, 0] * stride).max(), (anchors[i][:, 1] * stride).max()) * 1.5
        ax.set_xlim(0, max_size)
        ax.set_ylim(0, max_size)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.grid(True)

        for anchor in anchors[i]:
            width = anchor[0] * stride
            height = anchor[1] * stride
            rect = patches.Rectangle((max_size / 3, max_size / 3), width, height,
                                     linewidth=2, edgecolor='b', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.show()

# 保存并显示图像
save_path = "anchor_boxes_visualization_fixed_square.png"
plot_anchors(anchors, strides, save_path)
