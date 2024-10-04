# %%

import keras.layers
import keras_cv
import tensorflow as tf
import tensorflow.image

dct_A_8 = tf.constant([
    [0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339,
     0.35355339, 0.35355339, 0.35355339],
    [0.49039264, 0.41573481, 0.27778512, 0.09754516, -0.09754516,
     -0.27778512, -0.41573481, -0.49039264],
    [0.46193977, 0.19134172, -0.19134172, -0.46193977, -0.46193977,
     -0.19134172, 0.19134172, 0.46193977],
    [0.41573481, -0.09754516, -0.49039264, -0.27778512, 0.27778512,
     0.49039264, 0.09754516, -0.41573481],
    [0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339,
     -0.35355339, -0.35355339, 0.35355339],
    [0.27778512, -0.49039264, 0.09754516, 0.41573481, -0.41573481,
     -0.09754516, 0.49039264, -0.27778512],
    [0.19134172, -0.46193977, 0.46193977, -0.19134172, -0.19134172,
     0.46193977, -0.46193977, 0.19134172],
    [0.09754516, -0.27778512, 0.41573481, -0.49039264, 0.49039264,
     -0.41573481, 0.27778512, -0.09754516]
])

def apply_2d_dct_to_nhwc_image(image):
    """
    在 NHWC 格式的图像上应用 2D DCT，对每个通道进行操作。

    参数:
    - image (tensor): 输入图像。

    返回:
    - tensor: DCT 变换后的图像。
    """
    return dct_A_8 @ image @ tf.transpose(dct_A_8)

DCTLayer = keras.layers.Lambda(lambda x: apply_2d_dct_to_nhwc_image(x))
IDCtLayer = keras.layers.Lambda(lambda x: apply_2d_idct_to_nhwc_image(x))

def apply_2d_idct_to_nhwc_image(dct_image):
    """
    在 NHWC 格式的 DCT 变换后的图像上应用 2D IDCT。

    参数:
    - dct_image (tensor): DCT 变换后的图像。

    返回:
    - tensor: IDCT 变换后的图像。
    """
    return tf.transpose(dct_A_8) @ dct_image @ dct_A_8

def rgb_to_ycbcr(rgb_image):
    """
    将 RGB 图像转换为 YCbCr。

    参数:
    - rgb_image (tensor): RGB 图像，值范围在 [0, 1]。

    返回:
    - tensor: YCbCr 图像，值范围在 [0, 1]。
    """
    matrix = tf.constant([
        [0.257, 0.504, 0.098],
        [-0.148, -0.291, 0.439],
        [0.439, -0.368, -0.071]
    ], dtype=tf.float32)

    shift = tf.constant([0.0625, 0.5, 0.5], dtype=tf.float32)
    shift = tf.constant([16, 128, 128], dtype=tf.float32)

    ycbcr_image = tf.tensordot(rgb_image, matrix, axes=[[-1], [-1]]) + shift
    ycbcr_image = tf.clip_by_value(ycbcr_image, 0.0, 255.0)
    return ycbcr_image

def ycbcr_to_rgb(ycbcr_image):
    """
    将 YCbCr 图像转换为 RGB。

    参数:
    - ycbcr_image (tensor): YCbCr 图像，值范围在 [0, 1]。

    返回:
    - tensor: RGB 图像，值范围在 [0, 1]。
    """
    matrix = tf.constant([
        [1.16414435389988, -0.00178889771362069, 1.59578620543534],
        [1.16414435389988, -0.391442764342237, -0.813482068507793],
        [1.16414435389988, 2.0178255096009, -0.0012458394791287]
    ], dtype=tf.float32)

    shift = tf.constant([-222.657965050778, 135.604068942406, -276.748507437984], dtype=tf.float32)

    rgb_image = tf.tensordot(ycbcr_image, matrix, axes=[[-1], [-1]]) + shift
    rgb_image = tf.clip_by_value(rgb_image, 0.0, 255.0)
    return rgb_image

jpeg_table = tf.constant([[[
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 130, 99]
    ],
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ],
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]
]]], dtype=tf.float32)

def repatchs(block_images, image_size=256, c=3, patch_size=128):
    """
    合并图像块，图像格式为 (N, Num_patch, C, Patch_size, Patch_size)，
    最终图像格式为 (N, H, W, C)。

    参数:
    - block_images (tensor): 图像块。
    - image_size (int): 图像尺寸。
    - c (int): 通道数。
    - patch_size (int): 分块大小。

    返回:
    - tensor: 合并后的图像。
    """
    patch_nums = image_size // patch_size
    re_images = tf.reshape(block_images, [-1, patch_nums, patch_nums, patch_size, patch_size, c])
    split_rows = tf.split(value=re_images, num_or_size_splits=patch_nums, axis=1)
    concat_rows = tf.concat(split_rows, axis=-3)
    split_cols = tf.split(value=concat_rows, num_or_size_splits=patch_nums, axis=2)
    concat_image = tf.concat(split_cols, axis=-2)
    image = tf.reshape(concat_image, [-1, image_size, image_size, c])
    return image

def patchs(image, patch_size=128, c=3, image_size=256):
    """
    将图像拆分成块。

    参数:
    - image (tensor): 输入图像。
    - patch_size (int): 分块大小。
    - c (int): 通道数。
    - image_size (int): 图像尺寸。

    返回:
    - tensor: 拆分后的图像块。
    """
    num_patch = image_size // patch_size
    patch_images = tf.image.extract_patches(image,
                                            [1, patch_size, patch_size, 1],
                                            [1, patch_size, patch_size, 1],
                                            [1, 1, 1, 1],
                                            "SAME")
    reshape_image = tf.reshape(patch_images, [-1, num_patch ** 2, patch_size, patch_size, c])
    return reshape_image

def patch_image(image, patch_size=8, c=3, image_size=128):
    """
    拆分图像块，输入格式为 (N, H, W, C)，
    返回格式为 (N, Num_patch, C, Patch_size, Patch_size)。

    参数:
    - image (tensor): 输入图像。
    - patch_size (int): 分块大小。
    - c (int): 通道数。
    - image_size (int): 图像尺寸。

    返回:
    - tensor: 拆分后的图像块。
    """
    num_patch = image_size // patch_size
    patch_images = tf.image.extract_patches(image,
                                            [1, patch_size, patch_size, 1],
                                            [1, patch_size, patch_size, 1],
                                            [1, 1, 1, 1],
                                            "SAME")
    reshape_image = tf.reshape(patch_images, [-1, num_patch ** 2, patch_size, patch_size, c])
    trans_image = tf.transpose(reshape_image, [0, 1, 4, 2, 3])
    return trans_image

def re_patch_image(block_images, image_size=128, c=3, patch_size=8):
    """
    合并图像块，图像格式为 (N, Num_patch, C, Patch_size, Patch_size)，
    最终图像格式为 (N, H, W, C)。

    参数:
    - block_images (tensor): 图像块。
    - image_size (int): 图像尺寸。
    - c (int): 通道数。
    - patch_size (int): 分块大小。

    返回:
    - tensor: 合并后的图像。
    """
    patch_nums = image_size // patch_size
    trans_image = tf.transpose(block_images, [0, 1, 3, 4, 2])
    re_images = tf.reshape(trans_image, [-1, patch_nums, patch_nums, patch_size, patch_size, c])
    split_rows = tf.split(value=re_images, num_or_size_splits=patch_nums, axis=1)
    concat_rows = tf.concat(split_rows, axis=-3)
    split_cols = tf.split(value=concat_rows, num_or_size_splits=patch_nums, axis=2)
    concat_image = tf.concat(split_cols, axis=-2)
    image = tf.reshape(concat_image, [-1, image_size, image_size, c])
    return image

def dct_drop(dct_image):
    """
    对 DCT 图像应用随机丢弃。

    参数:
    - dct_image (tensor): DCT 变换后的图像。

    返回:
    - tensor: 随机丢弃后的图像。
    """
    x = tf.linspace(0.0, 1.2, 8)
    y = tf.linspace(0.0, 1.2, 8)
    xx, yy = tf.meshgrid(x, y)
    gradient = (xx + yy) / 2
    rate = 0.4
    adjusted_gradient = gradient * rate
    random_matrix = tf.random.uniform((8, 8), minval=0, maxval=1)
    thresholded_matrix = tf.where(random_matrix < adjusted_gradient, 0.0, 1.0)
    return dct_image * thresholded_matrix

def jpegQ(image, q):
    """
    应用指定质量的 JPEG 压缩。

    参数:
    - image (tensor): 输入图像。
    - q (int): JPEG 压缩质量。

    返回:
    - tensor: 压缩后的图像。
    """
    scale = 2 - q * 0.02 if q >= 50 else 50 / q
    return jpeg(image, scale)

def jpeg(image, scale):
    """
    对图像应用 JPEG 压缩。

    参数:
    - image (tensor): 输入图像。
    - scale (float): 压缩比例。

    返回:
    - tensor: 压缩后的图像。
    """
    image = image * 255 + tf.random.uniform(tf.shape(image), minval=-1.0, maxval=1.0)
    jpeg_table_Q = jpeg_table * scale
    jpeg_table_Q = tf.math.round(jpeg_table_Q)
    jpeg_table_Q = tf.clip_by_value(jpeg_table_Q, 1.0, 10000)
    ycbcr = rgb_to_ycbcr(image)
    y = ycbcr[:, :, :, :1]
    cbcr = ycbcr[:, :, :, 1:]
    sam_cbcr = keras.layers.AveragePooling2D(pool_size=[2, 2], strides=[2, 2])(cbcr)
    patch_img_y = patch_image(y, 8, c=1)
    dct_image_y = apply_2d_dct_to_nhwc_image(patch_img_y)
    qu_dct_y = dct_image_y / jpeg_table_Q[:, :, :1, :, :]
    sim_dct_y = tf.where(abs(qu_dct_y) >= 0.5, qu_dct_y, qu_dct_y ** 3)
    drop_dct_y = dct_drop(sim_dct_y)
    sim_dct_y = drop_dct_y * jpeg_table_Q[:, :, :1, :, :]
    sim_idct_y = apply_2d_idct_to_nhwc_image(sim_dct_y)
    re_patch_y = re_patch_image(sim_idct_y, c=1)
    patch_img_c = patch_image(sam_cbcr, 8, c=2, image_size=64)
    dct_image_c = apply_2d_dct_to_nhwc_image(patch_img_c)
    qu_dct_c = dct_image_c / jpeg_table_Q[:, :, 1:, :, :]
    sim_dct_c = tf.where(abs(qu_dct_c) >= 0.5, qu_dct_c, qu_dct_c ** 3)
    drop_dct_c = dct_drop(sim_dct_c)
    sim_dct_c = drop_dct_c * jpeg_table_Q[:, :, 1:, :, :]
    sim_idct_c = apply_2d_idct_to_nhwc_image(sim_dct_c)
    re_patch_c = re_patch_image(sim_idct_c, 64, 2)
    up_idct_c = keras.layers.UpSampling2D()(re_patch_c)
    re_patch = tf.concat([re_patch_y, up_idct_c], axis=-1)
    rgb_image = ycbcr_to_rgb(re_patch)
    return rgb_image / 255

def load_and_preprocess_image(image_path):
    """
    加载并预处理图像。

    参数:
    - image_path (str): 图像路径。

    返回:
    - tensor: 预处理后的图像。
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

class JPEGAttack(keras_cv.layers.BaseImageAugmentationLayer):
    """
    定义一个 JPEG 攻击层。
    """
    def augment_image(self, image, transformation, **kwargs):
        """
        对图像应用 JPEG 攻击。

        参数:
        - image (tensor): 输入图像。
        - transformation: 变换参数。

        返回:
        - tensor: 攻击后的图像。
        """
        q = tf.random.uniform((), minval=self.scale[0], maxval=self.scale[1])
        q = tf.math.round(q)
        shape = tf.shape(image)
        img = tf.expand_dims(image, 0)
        out_jpeg = jpeg(img, q)
        return tf.reshape(out_jpeg, shape)

    def __init__(self, Q, seed=None, **kwargs):
        """
        初始化 JPEG 攻击层。

        参数:
        - Q (list): 压缩质量范围。
        - seed: 随机种子。
        """
        super().__init__(seed, **kwargs)
        min_s = 2 - Q[0] * 0.02 if Q[0] >= 50 else 50 / Q[0]
        max_s = 2 - Q[1] * 0.02 if Q[1] >= 50 else 50 / Q[1]
        self.scale = [min_s, max_s]
        self.average_pool = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        self.up_sample = keras.layers.UpSampling2D()


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    img = load_and_preprocess_image('datasets/COCO128/000000574232.jpg')
    img = tf.expand_dims(img, 0)
    img = tf.image.resize(img, [128, 128])

    """
    该函数用于测试对图像进行分块和重新分块的功能。
    它将输入的图像分成更小的块，对其进行重塑，然后重新组合。
    最后，它将显示原始图像、重塑的块和重新组合的图像。

    参数:
    img (tf.Tensor): 要处理的输入图像。其形状应为 (N, H, W, C)，其中 N 为图像的数量，
                      H 为高度，W 为宽度，C 为颜色通道的数量。
    返回:
    None. 该函数仅显示图像。
    """
    def test_patch_repatch():

        # img 形状： N，H, W, C
        patchs = patch_image(img, 8)  # 分块图像尺寸 N， 分块数， C， patch_size, patch_size
        patch_reshape = tf.reshape(patchs, [1, 16, 16, 3, 8, 8])
        ind = 1
        for i in patch_reshape[0]:
            for j in i:
                plt.subplot(16, 16, ind)
                ind += 1
                img2 = tf.transpose(j, [1, 2, 0])
                plt.imshow(img2)
                plt.axis('off')
        plt.show()
        re_patch = re_patch_image(patchs)  # 合并后图像尺寸 N，H，W，C

        plt.imshow(re_patch[0])
        plt.show()
        plt.imshow(img[0])
        plt.show()


    test_patch_repatch()
    raise NotImplementedError()

    train_img = tf.Variable(initial_value=img, trainable=True)
    with tf.GradientTape() as tape:
        o_jpeg = JPEGAttack([30, 100])(train_img)
        loss = o_jpeg
    grad = tape.gradient(loss, train_img)
    print(grad)
    plt.imshow(o_jpeg[0])
    plt.show()
    psnr = tf.image.psnr(img, o_jpeg, 1.0)
    print(psnr)

    import sympy

    r, g, b, y, cr, cb = sympy.symbols('r g b y cr cb')
    f1 = 0.257 * r + 0.504 * g + 0.098 * b + 16 - y
    f2 = -0.148 * r - 0.291 * g + 0.439 * b + 128 - cb
    f3 = 0.439 * r - 0.368 * g - 0.071 * b + 128 - cr

    sol = sympy.solve([f1, f2, f3], [r, g, b])
