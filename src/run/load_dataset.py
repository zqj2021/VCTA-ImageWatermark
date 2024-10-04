import tensorflow as tf

import train_config
from train_config import *

class ImageDataset:
    """
    图像数据集类，用于加载和处理图像和水印数据。
    """
    def __init__(self, _image=train_config.train_image, _water=train_config.train_water, load_path=None, shuffle=True,
                 batch_size2=1, image_size2=image_size, water_size2=water_size):
        """
        初始化图像数据集类。

        参数:
        - _image (str): 图像数据目录。
        - _water (str): 水印数据目录。
        - load_path (str): 数据加载路径。
        - shuffle (bool): 是否打乱数据。
        - batch_size2 (int): 批量大小。
        - image_size2 (int): 图像尺寸。
        - water_size2 (int): 水印尺寸。
        """
        if load_path is None:
            # 加载水印数据集
            waters = tf.keras.utils.image_dataset_from_directory(
                directory=dataset_base_path + _water,
                image_size=[water_size2, water_size2],
                batch_size=batch_size2,
                label_mode=None,
                color_mode='grayscale',
                interpolation='mitchellcubic',
                crop_to_aspect_ratio=True,
                shuffle=shuffle,
            )
            # 加载图像数据集
            images_int = tf.keras.utils.image_dataset_from_directory(
                directory=dataset_base_path + _image,
                image_size=[image_size2, image_size2],
                batch_size=batch_size2,
                label_mode=None,
                color_mode='rgb',
                interpolation='mitchellcubic',
                crop_to_aspect_ratio=True,
                shuffle=shuffle,
            )
            # 将水印数据二值化
            self.waters_binary1: tf.data.Dataset = waters.map(lambda x: tf.where(x > 128, 1.0, 0.0))
            # 将图像数据归一化
            self.images1: tf.data.Dataset = images_int.map(lambda x: x / 255.0)
        else:
            # 从指定路径加载数据
            self.images1, self.waters_binary1 = self.load(load_path)
        self.waters = iter(self.waters_binary1.repeat())
        self.images = iter(self.images1.repeat())
        # 创建数据集生成器
        self.dataset = (tf.data.Dataset
                        .from_generator(generator=self.generate,
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=((1, image_size, image_size, 3), (1, water_size, water_size, 1)))
                        .unbatch().batch(batch_size))
        self.iter = iter(self.generate())

    def save(self, path):
        """
        保存数据集到指定路径。

        参数:
        - path (str): 保存路径。
        """
        self.images1.save(f"{path}/images", compression='GZIP')
        self.waters_binary1.save(f"{path}/waters", compression='GZIP')

    def load(self, load_path=None):
        """
        从指定路径加载数据集。

        参数:
        - load_path (str): 加载路径。

        返回:
        - tuple: 图像数据集和水印数据集。
        """
        images1 = tf.data.Dataset.load(f"{load_path}/images")
        waters_binary1 = tf.data.Dataset.load(f"{load_path}/waters")
        return images1, waters_binary1

    def getStndataset(self):
        """
        获取标准化数据集。

        返回:
        - tf.data.Dataset: 标准化数据集。
        """
        return self.dataset.map(lambda x, y: (x, None))

    def generate(self):
        """
        数据生成器。

        返回:
        - generator: 生成图像和水印数据对的生成器。
        """
        while True:
            image = next(self.images)
            water = next(self.waters)
            yield image, water

    def get_train_dataset(self):
        """
        获取训练数据集。

        返回:
        - tf.data.Dataset: 训练数据集。
        """
        return self.dataset.map(lambda x, y: ((x, y), None))

    def take(self):
        """
        获取下一个数据对。

        返回:
        - tuple: 图像和水印数据对。
        """
        return next(self.iter)

if __name__ == '__main__':
    # train_dataset = ImageDataset()
    test_dataset = ImageDataset(_image=train_config.test_image, _water=train_config.test_water)
    test_dataset.save("./datasets/test_data")
