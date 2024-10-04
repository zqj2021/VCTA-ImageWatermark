import keras.layers
import keras_cv.layers
import tensorflow as tf
import numpy as np

import jpeg_attack
import train_config

class AddGaussNoise(keras.layers.Layer):
    """
    定义一个添加高斯噪声的层。
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """
        对输入添加高斯噪声。

        参数:
        - inputs (tensor): 输入张量。

        返回:
        - tensor: 添加噪声后的张量。
        """
        rate = tf.random.uniform([], minval=0.0, maxval=1.0)
        rate = tf.round(rate)
        factor = tf.random.uniform([], 0.0, 0.1, dtype=tf.float32)
        noise = tf.random.normal(tf.shape(inputs))
        return tf.clip_by_value(rate * factor * noise + inputs, 0, 1)


from scipy.ndimage import median_filter

class MedianFilterLayer(keras_cv.layers.BaseImageAugmentationLayer):
    """
    定义一个中值滤波层。
    """
    def __init__(self, **kwargs):
        super(MedianFilterLayer, self).__init__(**kwargs)
        self.filter_size = tf.constant([1, 3, 5, 7])

    def augment_image(self, image, transformation, **kwargs):
        """
        对图像应用中值滤波。

        参数:
        - image (tensor): 输入图像。
        - transformation: 变换参数。

        返回:
        - tensor: 滤波后的图像。
        """
        filter = tf.random.shuffle(self.filter_size)[0]
        filtered_image = tf.numpy_function(self.apply_median_filter, [image, filter], tf.float32)
        filtered_image.set_shape(image.get_shape())
        return filtered_image

    def apply_median_filter(self, image, filter_size):
        """
        应用中值滤波。

        参数:
        - image (numpy.ndarray): 输入图像。
        - filter_size (int): 滤波器大小。

        返回:
        - numpy.ndarray: 滤波后的图像。
        """
        image = image.numpy() if hasattr(image, 'numpy') else image
        filtered_image = median_filter(image, size=(filter_size, filter_size, 1))
        return filtered_image

    def get_config(self):
        """
        获取配置。

        返回:
        - dict: 配置信息。
        """
        config = super(MedianFilterLayer, self).get_config()
        config.update({
            'filter_size': self.filter_size
        })
        return config

class LamdaLayer(keras.layers.Layer):
    """
    定义一个Lambda层，可以应用自定义函数。
    """
    def __init__(self, fun=None, **kwargs):
        super().__init__(**kwargs)
        self.fun = fun

    def call(self, inputs, *args, **kwargs):
        """
        调用自定义函数。

        参数:
        - inputs (tensor): 输入张量。

        返回:
        - tensor: 处理后的张量。
        """
        return self.fun(self, inputs)

class FastDropout(keras.layers.Layer):
    """
    定义一个快速Dropout层。
    """
    def __init__(self, min_rate=0.0, max_rate=0.7, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.max_rate = max_rate
        self.min_rate = min_rate
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        """
        对输入应用Dropout。

        参数:
        - inputs (tensor): 输入张量。

        返回:
        - tensor: 处理后的张量。
        """
        rate = tf.random.uniform([], self.min_rate, self.max_rate)
        random_uni = tf.random.uniform(tf.shape(inputs)[:-1], 0, 1)
        mask = random_uni > rate
        return tf.expand_dims(tf.cast(mask, tf.float32), -1) * inputs

class RandomDropout(keras_cv.layers.BaseImageAugmentationLayer):
    """
    定义一个随机Dropout层。
    """
    def __init__(self, drop_rate=0.3, seed=None, **kwargs):
        super().__init__(seed, **kwargs)
        self.dropout = drop_rate

    def augment_image(self, image, transformation, **kwargs):
        """
        对图像应用Dropout。

        参数:
        - image (tensor): 输入图像。
        - transformation: 变换参数。

        返回:
        - tensor: 处理后的图像。
        """
        rate = tf.random.uniform([], 0, self.dropout)
        random_uni = tf.random.uniform(tf.shape(image)[:-1], 0, 1)
        mask = random_uni > rate
        return tf.expand_dims(tf.cast(mask, tf.float32), -1) * image

class RandoGaussNoise(keras_cv.layers.BaseImageAugmentationLayer):
    """
    定义一个随机高斯噪声层。
    """
    def __init__(self, factor=0.1, clip=True, seed=None, **kwargs):
        super().__init__(seed, **kwargs)
        self.factor = factor
        self.clip = clip

    def augment_image(self, image, transformation, **kwargs):
        """
        对图像应用高斯噪声。

        参数:
        - image (tensor): 输入图像。
        - transformation: 变换参数。

        返回:
        - tensor: 处理后的图像。
        """
        random_fac = tf.random.uniform([], 0.0, self.factor)
        noise = tf.random.normal(shape=tf.shape(image), stddev=random_fac)
        noise_image = noise + image
        if self.clip:
            return tf.clip_by_value(noise_image, 0.0, 1.0)
        return noise_image

class SaltPepperNoiseLayer(keras_cv.layers.BaseImageAugmentationLayer):
    """
    定义一个椒盐噪声层。
    """
    def __init__(self, noise_ratio=0.05, **kwargs):
        super(SaltPepperNoiseLayer, self).__init__(**kwargs)
        self.noise_ratio = noise_ratio

    def augment_image(self, image, transformation, **kwargs):
        """
        对图像应用椒盐噪声。

        参数:
        - image (tensor): 输入图像。
        - transformation: 变换参数。

        返回:
        - tensor: 处理后的图像。
        """
        shape = tf.shape(image)
        height, width, channels = shape[0], shape[1], shape[2]
        noise_mask = tf.random.uniform([height, width, 1], minval=0., maxval=1.) < self.noise_ratio
        noise_values = tf.cast(tf.random.uniform([height, width, 1], minval=0, maxval=2), image.dtype)
        noise_mask = tf.tile(noise_mask, [1, 1, channels])
        noise_values = tf.tile(noise_values, [1, 1, channels])
        image = tf.where(noise_mask, noise_values, image)
        return image

    def get_config(self):
        """
        获取配置。

        返回:
        - dict: 配置信息。
        """
        config = super(SaltPepperNoiseLayer, self).get_config()
        config.update({
            'noise_ratio': self.noise_ratio
        })
        return config

class SaltPepperNoiseColor(tf.keras.layers.Layer):
    """
    定义一个彩色椒盐噪声层。
    """
    def __init__(self, noise_ratio=0.05, **kwargs):
        super(SaltPepperNoiseColor, self).__init__(**kwargs)
        self.noise_ratio = noise_ratio

    def call(self, inputs):
        """
        对输入应用彩色椒盐噪声。

        参数:
        - inputs (tensor): 输入张量。

        返回:
        - tensor: 处理后的张量。
        """
        shape = tf.shape(inputs)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        dtype = inputs.dtype
        noise_mask = tf.random.uniform([batch_size, height, width, 1], minval=0., maxval=1., dtype=dtype) < self.noise_ratio
        noise_values = tf.cast(tf.random.uniform([batch_size, height, width, 1], minval=0, maxval=2, dtype=tf.int32), dtype)
        noise_values_repeated = tf.tile(noise_values, [1, 1, 1, channels])
        noise_mask_repeated = tf.tile(noise_mask, [1, 1, 1, channels])
        outputs = tf.where(noise_mask_repeated, noise_values_repeated, inputs)
        return outputs

    def get_config(self):
        """
        获取配置。

        返回:
        - dict: 配置信息。
        """
        config = super(SaltPepperNoiseColor, self).get_config()
        config.update({'noise_ratio': self.noise_ratio})
        return config

augmentation_layers = {
    'gauss_noise': RandoGaussNoise(0.10),
    'jpeg': keras_cv.layers.RandomJpegQuality((25, 100)),
    'sim_jpeg': jpeg_attack.JPEGAttack((25, 100)),
    'gauss_blur': keras_cv.layers.RandomGaussianBlur((5, 5), (1.0, 1.0)),
    'crop': keras_cv.layers.RandomCutout((0.3, 1.0), (0.3, 1.0)),
    'dropout': RandomDropout(0.7),
    'med': MedianFilterLayer(),
    'salt': SaltPepperNoiseLayer(0.1)
}

def AttackLayer(rate=0.8):
    """
    根据配置返回攻击层。

    参数:
    - rate (float): 攻击率。

    返回:
    - keras层: 攻击层。
    """
    if train_config.attack_method != 'all':
        return augmentation_layers[train_config.attack_method]
    return keras_cv.layers.RandomAugmentationPipeline(layers=augmentation_layers.values(),
                                                      augmentations_per_image=1,
                                                      rate=rate, name="AttackLayer")

class BatchAttackLayer(keras.layers.Layer):
    """
    定义一个批量攻击层。
    """
    def __init__(self, rate, layers=augmentation_layers.values(), **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.random_choice = keras_cv.layers.RandomChoice(layers, batchwise=True)

    def call(self, inputs, training=None, *args, **kwargs):
        """
        对输入应用批量攻击。

        参数:
        - inputs (tensor): 输入张量。
        - training (bool): 是否在训练模式下。

        返回:
        - tensor: 处理后的张量。
        """
        if training:
            random_value = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
            out = tf.cond(tf.less(random_value, self.rate), lambda: self.random_choice(inputs), lambda: inputs)
            return out
        else:
            return inputs

if __name__ == '__main__':
    input_feature = tf.random.normal([batch_size, xx, xx])
    input_feature = tf.Variable(initial_value=input_feature, trainable=True)
    k_cc = tf.Variable(initial_value=[0, 2, 3, 4, ..., ], trainable=True)
    with tf.GradientTape() as tape:
        out = get_k_means_dis(input_feature, k_cc)
        loss = cal_loss_by_k_dis(out)
    gradient = tape.gradient(loss, input)
    print(gradient)
