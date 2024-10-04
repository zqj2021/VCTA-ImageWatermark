import io
import keras.callbacks
import keras.layers
import keras_cv
import tensorflow as tf
from matplotlib import pyplot as plt

import attacks_layer
import load_dataset
import train_config

class ShowImageCallback(keras.callbacks.Callback):
    """
    定义一个回调函数，在训练过程中显示和保存图像。
    """
    def __init__(self, dataset: load_dataset.ImageDataset):
        """
        初始化回调函数。

        参数:
        - dataset (ImageDataset): 用于生成图像的数据集。
        """
        super().__init__()
        self.best = float('inf')
        self.modelExtr = None
        self.modelEmbed: keras.models.Model = None
        self.modelAttack = None
        self.model : keras.models.Model
        self.images = iter(dataset.dataset)
        self.writer = tf.summary.create_file_writer(train_config.log_dir)

    @staticmethod
    def plot_to_image(figure):
        """
        将 matplotlib 图转换为 PNG 图像并返回。

        参数:
        - figure: matplotlib 图。

        返回:
        - tensor: PNG 图像。
        """
        # 将图像保存为 PNG 格式到内存中
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # 关闭图像以防止显示
        plt.close(figure)
        buf.seek(0)
        # 将 PNG 缓冲区转换为 TF 图像
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # 添加批次维度
        image = tf.expand_dims(image, 0)
        return image

    def on_train_begin(self, logs=None):
        """
        在训练开始时调用，初始化模型和攻击层。
        """
        self.modelAttack = attacks_layer.AttackLayer(rate=1.0)
        self.modelEmbed = self.model.get_layer("embed_model")
        self.modelEmbed.summary()
        self.model: keras.models.Model
        self.modelExtr: keras.models.Model = self.model.get_layer('extr_model')
        self.modelExtr.summary()

    def on_epoch_end(self, epoch, logs=None):
        """
        在每个训练周期结束时调用，保存最佳模型。

        参数:
        - epoch (int): 当前训练周期。
        - logs (dict): 日志数据。
        """
        cur_ber = logs['val_ber']
        if cur_ber < 0.1 and cur_ber < self.best:
            print('------------------best ber (save model)--------------------')
            self.modelEmbed.save(train_config.embed_model_dir())
            self.modelExtr.save(train_config.extr_model_dir())
            self.best = cur_ber
            print(f'-----------------save epoch - {epoch} ---------------------')

    def on_epoch_begin2(self, epoch, _):
        """
        在每个训练周期开始时调用，显示和记录图像。

        参数:
        - epoch (int): 当前训练周期。
        """
        o_image, o_water = next(self.images)
        o_image = o_image[0:4]
        o_water = o_water[0:4]
        p_image = self.modelEmbed.predict([o_image, o_water])
        attack_image = self.modelAttack(p_image, training=True)
        p_water = self.modelExtr.predict([p_image])
        p_water_attack = self.modelExtr.predict([attack_image])
        p_data = list(zip(o_image, attack_image, o_water, p_water, p_water_attack))
        fig = plots_n(p_data, row=4, col=5)
        image = ShowImageCallback.plot_to_image(fig)
        fig.show()
        with self.writer.as_default():
            tf.summary.image("Train&Test", image, step=epoch)

def plots_n(data, row, col):
    """
    绘制多张图像。

    参数:
    - data (list): 图像数据列表。
    - row (int): 行数。
    - col (int): 列数。

    返回:
    - figure: matplotlib 图。
    """
    _fig, _ = plt.subplots(row, col)
    for i in range(row):
        for j in range(col):
            plt.subplot(row, col, i * col + j + 1)
            plt.axis('off')
            plt.imshow(data[i][j], cmap='gray')

    return _fig
