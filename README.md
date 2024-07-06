# 73-Pattern-recognition-course-design
2024年华科aia模式识别课程设计-73-真实环境海面目标识别
相关代码注释
数据集：
Jianqie.py
import os
import cv2

path1 = './newcoast'  # 生成的所有图片的路径
all_img_num = 5  # 图片名字的位数，例如：五位数 00001.jpg

videoname = 'target_new.mp4'  # 视频的名字
capture = cv2.VideoCapture(videoname)

# 检查目录是否存在，不存在则创建
if not os.path.exists(path1):
    os.makedirs(path1)

num = 4050  # 初始编号
if capture.isOpened():
    while True:
        ret, img = capture.read()  # img 就是一帧图片
        
        if not ret:
            break  # 当获取完最后一帧就结束
        
        # 生成图片名称，确保名称格式一致
        name = f"{path1}/{str(num).zfill(all_img_num)}.jpg"
        print(name)
        
        # 保存图片
        cv2.imwrite(name, img)
        num += 1

        # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
        # cv2.imshow('Frame', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
else:
    print('视频打开失败！')

# 释放视频捕获对象
capture.release()
# cv2.destroyAllWindows()  # 如果使用了 cv2.imshow() 则需要销毁窗口
Huafen.py
import shutil
import random
from pathlib import Path

# 输入数据集路径
images_path = Path('./new/images')  # 替换为实际图片文件夹路径
labels_path = Path('./new/labels')  # 替换为实际标签文件夹路径

# 输出数据集路径
output_path = Path('new_split')

# 创建输出目录结构
(output_path / 'train' / 'images').mkdir(parents=True, exist_ok=True)
(output_path / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
(output_path / 'val' / 'images').mkdir(parents=True, exist_ok=True)
(output_path / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

# 获取所有图片和标签文件名（假设图片是 .jpg 格式，标签是 .txt 格式）
images = sorted(images_path.glob('*.jpg'))
labels = sorted(labels_path.glob('*.txt'))

# 确保图片和标签一一对应
assert len(images) == len(labels), "图片和标签数量不匹配。"
for img, lbl in zip(images, labels):
    assert img.stem == lbl.stem, f"图片和标签名称不匹配：{img} 和 {lbl}"

# 打乱顺序
data = list(zip(images, labels))
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(data)  # 打乱数据顺序

# 计算训练集数量
train_ratio = 0.8  # 训练集比例
train_count = int(len(data) * train_ratio)

# 分别复制到训练集和验证集
for i, (img, lbl) in enumerate(data):
    if i < train_count:
        shutil.copy(img, output_path / 'train' / 'images' / img.name)  # 复制图片到训练集
        shutil.copy(lbl, output_path / 'train' / 'labels' / lbl.name)  # 复制标签到训练集
    else:
        shutil.copy(img, output_path / 'val' / 'images' / img.name)  # 复制图片到验证集
        shutil.copy(lbl, output_path / 'val' / 'labels' / lbl.name)  # 复制标签到验证集

print("数据集拆分完成。")


Sea_5s.yaml
nc: 3 # 类别数量
depth_multiple: 0.33 # 模型深度倍数
width_multiple: 0.50 # 层通道倍数

# 检测头的锚点
anchors:
  - [0.9531, 1.1440, 1.4713, 1.4238, 3.1402, 2.2344] # P3/8
  - [4.8121, 1.9816, 8.3975, 2.8052, 21.8237, 2.2261] # P4/16
  - [18.6426, 1.4270, 19.4326, 2.3723, 18.0818, 4.5073] # P5/32

backbone:
  # [来源, 数量, 模块, 参数]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2, 卷积层, 64个卷积核, 核大小6, 步幅2, 填充2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4, 卷积层, 128个卷积核, 核大小3, 步幅2
    [-1, 3, C3, [128]], # 2, C3模块, 128个卷积核, 重复3次
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8, 卷积层, 256个卷积核, 核大小3, 步幅2
    [-1, 6, C3, [256]], # 4, C3模块, 256个卷积核, 重复6次
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16, 卷积层, 512个卷积核, 核大小3, 步幅2
    [-1, 9, C3, [512]], # 6, C3模块, 512个卷积核, 重复9次
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32, 卷积层, 1024个卷积核, 核大小3, 步幅2
    [-1, 3, C3, [1024]], # 8, C3模块, 1024个卷积核, 重复3次
    [-1, 1, SPPF, [1024, 5]], # 9, SPPF模块, 1024个卷积核, 核大小5
  ]

head: [
    [-1, 1, Conv, [512, 1, 1]], # 10, 卷积层, 512个卷积核, 核大小1, 步幅1
    [-1, 1, nn.Upsample, [None, 2, "nearest"]], # 11, 上采样层, 缩放因子2, 最近邻插值
    [[-1, 6], 1, Concat, [1]], # 12, 拼接层, 与层6拼接
    [-1, 3, C3, [512, False]], # 13, C3模块, 512个卷积核, 重复3次, 无捷径

    [-1, 1, Conv, [256, 1, 1]], # 14, 卷积层, 256个卷积核, 核大小1, 步幅1
    [-1, 1, nn.Upsample, [None, 2, "nearest"]], # 15, 上采样层, 缩放因子2, 最近邻插值
    [[-1, 4], 1, Concat, [1]], # 16, 拼接层, 与层4拼接
    [-1, 3, C3, [256, False]], # 17, C3模块, 256个卷积核, 重复3次, 无捷径 (P3/8-小)

    [-1, 1, Conv, [256, 3, 2]], # 18, 卷积层, 256个卷积核, 核大小3, 步幅2
    [[-1, 14], 1, Concat, [1]], # 19, 拼接层, 与层14拼接
    [-1, 3, C3, [512, False]], # 20, C3模块, 512个卷积核, 重复3次, 无捷径 (P4/16-中)

    [-1, 1, Conv, [512, 3, 2]], # 21, 卷积层, 512个卷积核, 核大小3, 步幅2
    [[-1, 10], 1, Concat, [1]], # 22, 拼接层, 与层10拼接
    [-1, 3, C3, [1024, False]], # 23, C3模块, 1024个卷积核, 重复3次, 无捷径 (P5/32-大)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # 24, 检测层, 使用层17, 20, 23, 类别数量和锚点
  ]


Data.yaml
path: datasets/train # 数据集根目录
train: images/train # 训练图像（相对于 'path'）
val: images/val # 验证图像（相对于 'path'）
test: images/test # 测试图像（可选）

# 类别
names:
  0: this # 类别0
  1: barrier # 类别1
  2: coast # 类别2

Anchor框：
Autoanchor.py
import random
import numpy as np
import torch
import yaml
from tqdm import tqdm
from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr("AutoAnchor: ")

def check_anchor_order(m):
    """检查并在必要时根据YOLOv5 Detect()模块中的步幅调整锚点顺序。"""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # 每个输出层的平均锚点面积
    da = a[-1] - a[0]  # 面积差
    ds = m.stride[-1] - m.stride[0]  # 步幅差
    if da and (da.sign() != ds.sign()):  # 顺序不一致
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)

@TryExcept(f"{PREFIX}ERROR")
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """评估锚点对数据集的适配性，并在必要时进行调整，支持自定义阈值和图像大小。"""
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 数据增强的缩放
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # 宽高

    def metric(k):  # 计算指标
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # 比例指标
        best = x.max(1)[0]  # 最佳x
        aat = (x > 1 / thr).float().sum(1).mean()  # 超过阈值的锚点
        bpr = (best > 1 / thr).float().mean()  # 最佳可能召回率
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # 模型步幅
    anchors = m.anchors.clone() * stride  # 当前锚点
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    if bpr > 0.98:  # 阈值
        LOGGER.info(f"{s}当前锚点适合数据集 ✅")
    else:
        LOGGER.info(f"{s}锚点不适合数据集 ⚠️，尝试改进...")
        na = m.anchors.numel() // 2  # 锚点数量
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # 替换锚点
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # 必须在像素空间（非网格空间）
            m.anchors /= stride
            s = f"{PREFIX}完成 ✅（可选：更新模型 *.yaml 以在未来使用这些锚点）"
        else:
            s = f"{PREFIX}完成 ⚠️（原始锚点优于新锚点，继续使用原始锚点）"
        LOGGER.info(s)

def kmean_anchors(dataset="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    从训练数据集创建K均值演化锚点。

    参数：
        dataset: data.yaml的路径或已加载的数据集
        n: 锚点数量
        img_size: 用于训练的图像大小
        thr: 锚点-标签宽高比阈值超参数 hyp['anchor_t']，默认=4.0
        gen: 使用遗传算法演化锚点的代数
        verbose: 打印所有结果

    返回：
        k: K均值演化锚点

    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # 计算指标
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # 比例指标
        return x, x.max(1)[0]  # x, 最佳x

    def anchor_fitness(k):  # 变异适应度
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # 适应度

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # 从小到大排序
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # 最佳可能召回率, 超过阈值的锚点
        s = (
            f"{PREFIX}thr={thr:.2f}: {bpr:.4f} 最佳可能召回率, {aat:.2f} 锚点超过阈值\n"
            f"{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-平均/最佳, "
            f"超过阈值={x[x > thr].mean():.3f}-平均: "
        )
        for x in k:
            s += "%i,%i, " % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml 文件
        with open(dataset, errors="ignore") as f:
            data_dict = yaml.safe_load(f)  # 模型字典
        from utils.dataloaders import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(data_dict["train"], augment=True, rect=True)

    # 获取标签宽高
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 宽高

    # 过滤
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f"{PREFIX}WARNING ⚠️ 发现极小的物体：{len(wh0)}个标签中的{i}个小于3个像素")
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # 过滤大于2个像素的
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # 乘以随机缩放0-1

    # K均值初始化
    try:
        LOGGER.info(f"{PREFIX}对{len(wh)}个点运行{k}个锚点的K均值...")
        assert n <= len(wh)  # 应用过度约束
        s = wh.std(0)  # 标准差用于白化
        k = kmeans(wh / s, n, iter=30)[0] * s  # 点
        assert n == len(k)  # K均值可能返回少于请求的点，如果wh不足或过于相似
    except Exception:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ 从K均值切换到随机初始化")
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # 随机初始化
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # 绘图
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # 点, 平均距离
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7),


Testanchor.py
import numpy as np

from gen_anchors import Model, Dataset, get_yolov5_data
from autoanchor import check_anchor_order, check_anchors

def test_dataset():
    """测试数据集和锚点检查功能。"""
    shapes, labels = get_yolov5_data("../datasets/", "train")  # 获取YOLOv5数据
    dataset = Dataset(shapes, labels)  # 创建数据集对象

    anchors = np.array([
        [10, 13, 16, 30, 33, 23],  # P3/8层的锚点
        [30, 61, 62, 45, 59, 119],  # P4/16层的锚点
        [116, 90, 156, 198, 373, 326]  # P5/32层的锚点
    ])
    stride = [8, 16, 32]  # 每层的步幅
    m = Model(anchors=anchors, stride=stride)  # 创建模型对象

    check_anchors(dataset, m)  # 检查锚点适配性
    # print('anchors:', m.anchors)

    return m  # 返回模型对象

def test_model(m=None):
    """测试模型的锚点顺序检查功能。"""
    if m is None:  # 如果未传入模型，则创建一个默认模型
        anchors = np.array([
            [10, 13, 16, 30, 33, 23],  # P3/8层的锚点
            [30, 61, 62, 45, 59, 119],  # P4/16层的锚点
            [116, 90, 156, 198, 373, 326]  # P5/32层的锚点
        ])
        stride = [8, 16, 32]  # 每层的步幅
        m = Model(anchors=anchors, stride=stride)  # 创建模型对象

    print("anchors:", m.anchors)  # 打印锚点
    print("stride:", m.stride)  # 打印步幅
    check_anchor_order(m)  # 检查并调整锚点顺序
    print("anchors:", m.anchors)  # 打印调整后的锚点
    print("stride:", m.stride)  # 打印步幅

if __name__ == '__main__':
    m = test_dataset()  # 运行数据集测试
    test_model(m)  # 运行模型测试


YOLO训练和预测
Train.py

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

def parse_opt(known=False):
    """解析用于YOLOv5训练、验证和测试的命令行参数。"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="初始权重路径")
    parser.add_argument("--cfg", type=str, default="", help="模型配置文件路径，例如 models/boat5s.yaml")
    parser.add_argument("--data", type=str, default=ROOT / "data/boat.yaml", help="数据集配置文件路径")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="超参数路径")
    parser.add_argument("--epochs", type=int, default=100, help="总训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="所有GPU的总批量大小，-1表示自动批量")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="训练和验证图像大小（像素）")
    parser.add_argument("--rect", action="store_true", help="矩形训练")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="恢复最近的训练")
    parser.add_argument("--nosave", action="store_true", help="仅保存最后的检查点")
    parser.add_argument("--noval", action="store_true", help="仅验证最后一个epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="禁用自动锚点")
    parser.add_argument("--noplots", action="store_true", help="不保存绘图文件")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="演化超参数x代")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="加载种群的位置"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="从上一代恢复演化")
    parser.add_argument("--bucket", type=str, default="", help="gsutil存储桶")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="图像缓存到ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="使用加权图像选择进行训练")
    parser.add_argument("--device", default="", help="cuda设备，例如0或0,1,2,3或cpu")
    parser.add_argument("--multi-scale", action="store_true", help="变换图像大小 +/- 50%")
    parser.add_argument("--single-cls", action="store_true", help="将多类数据训练为单类")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="优化器")
    parser.add_argument("--sync-bn", action="store_true", help="使用同步批标准化，仅在DDP模式下可用")
    parser.add_argument("--workers", type=int, default=1, help="最大数据加载器工作线程数（在DDP模式下每个RANK）")
    parser.add_argument("--project", default=ROOT / "runs/train", help="保存到项目/名称")
    parser.add_argument("--name", default="exp", help="保存到项目/名称")
    parser.add_argument("--exist-ok", action="store_true", help="项目/名称已存在则不增加编号")
    parser.add_argument("--quad", action="store_true", help="四倍数据加载器")
    parser.add_argument("--cos-lr", action="store_true", help="使用余弦学习率调度器")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="标签平滑系数")
    parser.add_argument("--patience", type=int, default=100, help="早停耐心（无改进轮数）")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="冻结层：主干=10，前3层=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="每x轮保存检查点（<1则禁用）")
    parser.add_argument("--seed", type=int, default=0, help="全局训练随机种子")
    parser.add_argument("--local_rank", type=int, default=-1, help="自动DDP多GPU参数，请勿修改")

    # 日志参数
    parser.add_argument("--entity", default=None, help="实体")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='上传数据，"val"选项')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="设置边界框图像日志记录间隔")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="使用的数据集版本别名")

    # NDJSON日志记录
    parser.add_argument("--ndjson-console", action="store_true", help="将ndjson记录到控制台")
    parser.add_argument("--ndjson-file", action="store_true", help="将ndjson记录到文件")

    return parser.parse_known_args()[0] if known else parser.parse_args()


detect.py
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

def parse_opt():
    """解析用于YOLOv5检测的命令行参数，设置推理选项和模型配置。"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "best.pt", help="模型路径或triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="文件/目录/URL/glob/screen/0(摄像头)")
    parser.add_argument("--data", type=str, default=ROOT / "data/boat.yaml", help="（可选）数据集配置文件路径")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="推理图像大小 h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--max-det", type=int, default=1000, help="每张图像的最大检测数")
    parser.add_argument("--device", default="", help="cuda设备，例如0或0,1,2,3或cpu")
    parser.add_argument("--view-img", action="store_true", help="显示结果")
    parser.add_argument("--save-txt", action="store_true", help="将结果保存为*.txt")
    parser.add_argument("--save-csv", action="store_true", help="以CSV格式保存结果")
    parser.add_argument("--save-conf", action="store_true", help="在--save-txt标签中保存置信度")
    parser.add_argument("--save-crop", action="store_true", help="保存裁剪的预测框")
    parser.add_argument("--nosave", action="store_true", help="不保存图像/视频")
    parser.add_argument("--classes", nargs="+", type=int, help="按类别过滤：--classes 0，或--classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="类别无关的NMS")
    parser.add_argument("--augment", action="store_true", help="增强推理")
    parser.add_argument("--visualize", action="store_true", help="可视化特征")
    parser.add_argument("--update", action="store_true", help="更新所有模型")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="将结果保存到项目/名称")
    parser.add_argument("--name", default="exp", help="将结果保存到项目/名称")
    parser.add_argument("--exist-ok", action="store_true", help="项目/名称已存在则不增加编号")
    parser.add_argument("--line-thickness", default=3, type=int, help="边界框线条粗细（像素）")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="隐藏标签")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="隐藏置信度")
    parser.add_argument("--half", action="store_true", help="使用FP16半精度推理")
    parser.add_argument("--dnn", action="store_true", help="使用OpenCV DNN进行ONNX推理")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频帧率步幅")
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展图像大小
    print_args(vars(opt))
    return opt

def print_args(args):
    """打印参数"""
    print('\n'.join(f'{k}: {v}' for k, v in args.items()))


segement.py
import torch
# 使用 torch.hub 从 pytorch/vision 库中加载预训练的 FCN-ResNet50 模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# 或者
# 使用 torch.hub 从 pytorch/vision 库中加载预训练的 FCN-ResNet101 模型
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
model.eval()  # 将模型设置为评估模式

# 样本执行（需要 torchvision 库）
from PIL import Image  # 导入 Pillow 库，用于图像处理
from torchvision import transforms  # 导入 torchvision 库中的 transforms 模块

# 打开输入图像
input_image = Image.open('00000.jpg')
input_image = input_image.convert("RGB")  # 将图像转换为 RGB 模式

# 定义预处理操作：转换为张量并标准化
preprocess = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

input_tensor = preprocess(input_image)  # 对图像进行预处理
input_batch = input_tensor.unsqueeze(0)  # 增加一个维度，创建一个 mini-batch，符合模型的输入要求

# 如果 GPU 可用，则将输入和模型移动到 GPU 上以提高速度
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# 关闭梯度计算，以加快推理速度
with torch.no_grad():
    output = model(input_batch)['out'][0]  # 获取模型输出，并取出 batch 的第一个元素
output_predictions = output.argmax(0)  # 对输出进行 argmax 操作，得到每个像素的类别预测

# 创建一个颜色调色板，为每个类别选择一种颜色
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])  # 定义颜色调色板
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette  # 为每个类别生成颜色
colors = (colors % 255).numpy().astype("uint8")  # 将颜色值限制在 0-255 范围内，并转换为 uint8 类型

# 绘制语义分割预测结果，为每个类别上色
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)  # 将预测结果转换为图像，并调整大小
r.putpalette(colors)  # 应用颜色调色板

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
plt.imshow(r)  # 显示分割结果图像
# plt.show()  # 显示图像窗口



预训练的deeplab模型进行测试代码：
import torch
# 使用 torch.hub 从 pytorch/vision 库中加载预训练的 DeepLabV3-ResNet50 模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# 或者以下任何一个变体
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()  # 将模型设置为评估模式

# 样本执行（需要 torchvision 库）
from PIL import Image  # 导入 Pillow 库，用于图像处理
from torchvision import transforms  # 导入 torchvision 库中的 transforms 模块

# 打开输入图像
input_image = Image.open('00000.jpg')
input_image = input_image.convert("RGB")  # 将图像转换为 RGB 模式

# 定义预处理操作：转换为张量并标准化
preprocess = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

input_tensor = preprocess(input_image)  # 对图像进行预处理
input_batch = input_tensor.unsqueeze(0)  # 增加一个维度，创建一个 mini-batch，符合模型的输入要求

# 如果 GPU 可用，则将输入和模型移动到 GPU 上以提高速度
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# 关闭梯度计算，以加快推理速度
with torch.no_grad():
    output = model(input_batch)['out'][0]  # 获取模型输出，并取出 batch 的第一个元素
output_predictions = output.argmax(0)  # 对输出进行 argmax 操作，得到每个像素的类别预测

# 创建一个颜色调色板，为每个类别选择一种颜色
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])  # 定义颜色调色板
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette  # 为每个类别生成颜色
colors = (colors % 255).numpy().astype("uint8")  # 将颜色值限制在 0-255 范围内，并转换为 uint8 类型

# 绘制语义分割预测结果，为每个类别上色
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)  # 将预测结果转换为图像，并调整大小
r.putpalette(colors)  # 应用颜色调色板

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
plt.imshow(r)  # 显示分割结果图像
# plt.show()  # 显示图像窗口

人机交互
Build.py
import os
import shutil

def build(main, method):
    parameters = ['nuitka',
                  '--standalone',  # 独立模式
                  '--mingw64',  # 强制使用MinGW64编译器
                  '--nofollow-imports',  # 不导入任何模块
                  '--plugin-enable=qt-plugins',  # 导入PyQt
                  '--follow-import-to=utils',  # 递归指定的模块或包
                  # '--windows-icon-from-ico=favicon.ico',  # 设置图标
                  ]
    if method == "0":
        path = os.path.join(os.getcwd(), output_dir, 'debug')
        parameters.append(f'--output-dir="{path}"')  # 指定最终文件的输出目录
    elif method == "1":
        path = os.path.join(os.getcwd(), output_dir, 'release')
        parameters.append("--windows-disable-console")  # 禁用控制台窗口
        # parameters.append("--windows-uac-admin")  # UAC
        parameters.append(f'--output-dir="{path}"')  # 指定最终文件的输出目录
    else:
        raise ValueError
    # nuitka
    os.system(f"{' '.join(parameters)} {main}.py")
    return os.path.join(path, main + '.dist', main + '.exe')

def movefile(src_path):
    dst_path = os.path.join(output_dir, 'publish')
    os.makedirs(dst_path, exist_ok=True)
    shutil.copy(src_path, os.path.join(dst_path, os.path.basename(src_path)))

if __name__ == "__main__":
    enter = 'Yolo2onnxDetectProjectDemo'
    output_dir = os.path.join(os.getcwd(), 'build_file')
    os.makedirs(output_dir, exist_ok=True)
    exe_path = build(enter, input("[Debug(0) / Release(1)]："))
    movefile(exe_path)
    # pass


