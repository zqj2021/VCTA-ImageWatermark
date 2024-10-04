import keras_cv
import tensorflow as tf
from scipy.ndimage import median_filter
from src.run import attacks_layer
import numpy as np
import random

def random_cutout(images, cutout_fraction):
    """
    对一批图像应用随机遮挡。

    参数:
    - images (tensor): 形状为 (N, H, W, C) 的四维张量，表示一批图像。
    - cutout_fraction (float): 遮挡尺寸相对于图像的比例。

    返回:
    - tensor: 形状与 'images' 相同的张量，应用了随机遮挡。
    """
    batch_size, height, width, channels = images.shape
    cutout_height = tf.cast(cutout_fraction * height, tf.int32)
    cutout_width = tf.cast(cutout_fraction * width, tf.int32)

    masks = []
    for _ in range(batch_size):
        top = tf.random.uniform([], minval=0, maxval=height - cutout_height, dtype=tf.int32)
        left = tf.random.uniform([], minval=0, maxval=width - cutout_width, dtype=tf.int32)

        mask = tf.ones((height, width, channels), dtype=images.dtype)
        paddings = [[top, height - cutout_height - top], [left, width - cutout_width - left], [0, 0]]
        mask = tf.pad(tf.zeros((cutout_height, cutout_width, channels), dtype=images.dtype), paddings,
                      constant_values=1)
        masks.append(mask)

    mask_batch = tf.stack(masks)
    images_with_cutout = images * mask_batch

    return images_with_cutout

def random_jpeg(images, quality):
    """
    对一批图像应用随机 JPEG 压缩。

    参数:
    - images (tensor): 输入图像。
    - quality (int): JPEG 压缩质量。

    返回:
    - tensor: 应用了随机 JPEG 压缩的图像。
    """
    res = list(map(lambda x: tf.image.adjust_jpeg_quality(x, quality), images))
    return np.array(res)

def random_gaussnoise(images, factors):
    """
    对一批图像应用随机高斯噪声。

    参数:
    - images (tensor): 输入图像。
    - factors (float): 噪声标准差。

    返回:
    - tensor: 应用了随机高斯噪声的图像。
    """
    return tf.random.normal(tf.shape(images), stddev=factors) + images

def random_mediaGlur(images, filter_size):
    """
    对一批图像应用随机中值滤波。

    参数:
    - images (tensor): 输入图像。
    - filter_size (int): 滤波器大小。

    返回:
    - tensor: 应用了随机中值滤波的图像。
    """
    res = list(map(lambda x: median_filter(x, size=(filter_size, filter_size, 1)), images))
    return np.array(res)

def random_papersalt(images, factors):
    """
    对一批图像应用随机椒盐噪声。

    参数:
    - images (tensor): 输入图像。
    - factors (float): 噪声比例。

    返回:
    - tensor: 应用了随机椒盐噪声的图像。
    """
    return attacks_layer.SaltPepperNoiseLayer(factors)(images)

def random_dropout(images, prob):
    """
    对一批图像应用随机 Dropout。

    参数:
    - images (tensor): 输入图像。
    - prob (float): Dropout 概率。

    返回:
    - tensor: 应用了随机 Dropout 的图像。
    """
    return attacks_layer.FastDropout(prob, prob)(images)

def random_gaussglur(images, factors):
    """
    对一批图像应用随机高斯模糊。

    参数:
    - images (tensor): 输入图像。
    - factors (float): 模糊因子。

    返回:
    - tensor: 应用了随机高斯模糊的图像。
    """
    return keras_cv.layers.RandomGaussianBlur((1, 1), factors)(images)

# 算法练习 与VIM操作的练习
# 归并排序的实现

a0 = np.random.normal(0, 0.1, size=[120])
a1 = [i for i in (a0 * 100).astype(np.int32)]
print(a1)

def merage_sort(arr):
    """
    归并排序算法。

    参数:
    - arr (list): 输入列表。

    返回:
    - list: 排序后的列表。
    """
    if len(arr) <= 1:
        return arr
    arro = []
    arr1 = merage_sort(arr[:len(arr) // 2])
    arr2 = merage_sort(arr[len(arr) // 2:])
    a1 = 0
    a2 = 0
    for i in range(len(arr)):
        if a1 < len(arr1) and (len(arr2) <= a2 or arr1[a1] < arr2[a2]):
            arro.append(arr1[a1])
            a1 += 1
        else:
            arro.append(arr2[a2])
            a2 += 1
    return arro

print(merage_sort(a1))

def findX(a1, p, q, x):
    """
    查找列表中的第 x 大元素。

    参数:
    - a1 (list): 输入列表。
    - p (int): 起始索引。
    - q (int): 结束索引。
    - x (int): 第 x 大元素的位置。

    返回:
    - int: 第 x 大的元素。
    """
    a = random.randint(p, q - 1)
    ax = a1[a]
    temp = a1[p]
    a1[p] = a1[a]
    a1[a] = temp
    j = q
    i = p + 1
    while j != i:
        if a1[i] > a1[p]:
            j -= 1
            t = a1[i]
            a1[i] = a1[j]
            a1[j] = t
        else:
            i += 1
    a1[p] = a1[i - 1]
    a1[i - 1] = ax
    if i - 1 == x:
        return ax
    elif i - 1 < x:
        return findX(a1, i, q, x)
    else:
        return findX(a1, p, i - 1, x)

print(findX(a1, 0, len(a1), 60))
