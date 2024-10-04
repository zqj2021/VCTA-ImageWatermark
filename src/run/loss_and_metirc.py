import keras.losses
import tensorflow as tf
from keras import layers

def calculate_batch_ber(y_true, y_pred, threshold=0.5):
    """
    使用矩阵运算计算批量中每个样本的误码率。

    参数:
    y_true -- 真实值的张量，预期为整型（0或1），形状为[batch_size, n]。
    y_pred -- 预测值的张量，为0到1之间的浮点数，与真实值同形状。
    threshold -- 用于将y_pred的浮点数值转换为0或1的阈值。

    返回:
    ber_batch -- 形状为[batch_size, 1]的张量，表示批量中每个样本的误码率。
    """
    # 将预测值根据阈值转换为0或1
    y_pred_int = tf.cast(tf.greater_equal(y_pred, threshold), tf.int32)
    y_true_int = tf.cast(y_true, tf.int32)

    # 计算每个样本的错误比特数
    errors = tf.reduce_sum(tf.abs(y_true_int - y_pred_int), axis=1)

    # 计算每个样本的总比特数
    total_bits = tf.shape(y_true)[1]  # 假设每个样本的比特数相同

    # 计算误码率
    ber_batch = tf.cast(errors, tf.float32) / tf.cast(total_bits, tf.float32 )

    # 重新整理形状以符合(batch_size, 1)
    ber_batch = tf.reshape(ber_batch, [-1, 1])

    return ber_batch

class RLoss(layers.Layer):
    """
    定义一个自定义损失层，计算均方误差损失并将其添加到总损失中。
    """
    def __init__(self, factor):
        super(RLoss, self).__init__()
        self.factor = factor

    def call(self, inputs, *args, **kwargs):
        """
        调用损失层。

        参数:
        - inputs: 输入张量。

        返回:
        - tensor: 输入张量。
        """
        basic_f, test1, test2, test3 = inputs

        def comp_loss(y_pred):
            dev = tf.losses.mean_squared_error(y_true=basic_f, y_pred=y_pred)
            mean_dev = tf.reduce_mean(dev)
            self.add_loss(mean_dev * self.factor)

        comp_loss(test1)
        comp_loss(test2)
        comp_loss(test3)
        return inputs

class LossLayer(layers.Layer):
    """
    定义一个损失层，计算图像和水印的损失。
    """
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, **_):
        """
        调用损失层。

        参数:
        - inputs: 输入张量。

        返回:
        - tuple: 处理后的图像和水印。
        """
        t_image, p_image, t_water, p_water = inputs
        f_t_water = keras.layers.Flatten()(t_water)
        f_p_water = keras.layers.Flatten()(p_water)
        extr_loss = keras.losses.binary_crossentropy(f_t_water, f_p_water)
        mse_loss = tf.losses.mean_squared_error(t_image, p_image)

        flatten_t = keras.layers.Flatten()(t_water)
        flatten_p = keras.layers.Flatten()(p_water)
        psnr = tf.image.psnr(t_image, p_image, 1.0)
        ber = calculate_batch_ber(flatten_t, flatten_p)
        mes_factor = tf.cond(tf.greater_equal(tf.reduce_mean(mse_loss), 2e-03), lambda: tf.ones_like(tf.reduce_mean(mse_loss)),  lambda :0.0)
        self.add_loss(tf.reduce_mean(extr_loss))
        # self.add_loss(tf.reduce_mean(mse_loss))

        self.add_metric(mse_loss, "mes-loss")
        self.add_metric(extr_loss, "bce-loss")
        self.add_metric(psnr, "psnr")
        self.add_metric(ber, "ber")
        return p_image, p_water

class LossStep(layers.Layer):
    """
    定义一个损失步骤层，计算噪声的均方误差损失。
    """
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):
        """
        调用损失步骤层。

        参数:
        - inputs: 输入张量。

        返回:
        - tensor: 处理后的图像。
        """
        r_noise, p_noise, image = inputs
        mes_loss = keras.losses.mean_squared_error(r_noise, p_noise)
        loss = tf.reduce_mean(mes_loss)
        self.add_loss(loss * 0.1)
        self.add_metric(loss, "noise_loss")
        return image
