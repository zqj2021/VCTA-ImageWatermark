import math
import keras
import keras_cv
import tensorflow as tf
import attacks_layer
import loss_and_metirc
import train_config
import u_net_block
import vit_emmbed
from train_config import image_size, water_size

def add_noise(image):
    """
    向图像添加随机高斯噪声。

    参数:
    - image (tensor): 输入图像。

    返回:
    - tensor: 添加噪声后的图像。
    """
    rate = tf.random.uniform([], minval=0.0, maxval=1.0)
    rate = tf.round(rate)
    factor = tf.random.uniform([], 0.0, 0.15, dtype=tf.float32)
    noise = tf.random.normal(image.shape)
    return tf.clip_by_value(rate * factor * noise, 0, 1)

def main_model(embed, extr):
    """
    创建主模型，包含嵌入和提取模型。

    参数:
    - embed (Model): 嵌入模型。
    - extr (Model): 提取模型。

    返回:
    - keras.Model: 主模型。
    """
    input_image = keras.layers.Input(shape=[image_size, image_size, 3])
    input_watermark = keras.layers.Input(shape=[water_size, water_size, 1])
    embed_image = embed([input_image, input_watermark])
    attack_image = attacks_layer.AttackLayer(rate=0.5)(embed_image)
    extr_water = extr(attack_image)
    embed_image, extr_water = loss_and_metirc.LossLayer()([input_image, embed_image, input_watermark, extr_water])
    return keras.models.Model(inputs=[input_image, input_watermark], outputs=[embed_image, extr_water])

def extr_model():
    """
    创建提取模型。

    返回:
    - keras.Model: 提取模型。
    """
    input_image = keras.layers.Input(shape=[image_size, image_size, 3])
    patch_embedding = keras_cv.layers.PatchingAndEmbedding(project_dim=512, patch_size=8)(input_image)
    vit_encoder = vit_emmbed.MulTransformerBlock(patch_embedding, 512, 4, 768, 0.3)
    vit_mid = keras.layers.Dense(768, activation=train_config.activation)(vit_encoder)
    vit_mid = keras.layers.Flatten()(vit_mid)
    drop_out_layer = keras.layers.Dropout(rate=0.1)(vit_mid)
    extr_water_out = keras.layers.Dense(water_size ** 2, activation='sigmoid')(drop_out_layer)
    extr_water = keras.layers.Reshape(target_shape=(water_size, water_size, -1))(extr_water_out)
    return keras.models.Model(inputs=[input_image], outputs=extr_water, name="extr_model")

def embed_model():
    """
    创建嵌入模型。

    返回:
    - keras.Model: 嵌入模型。
    """
    input_image = keras.layers.Input(shape=(image_size, image_size, 3))
    input_watermark = keras.layers.Input(shape=(water_size, water_size, 1))
    flatten_water = keras.layers.Flatten()(input_watermark)
    up_water = keras.layers.Dense(image_size ** 2, activation=train_config.activation)(flatten_water)
    up_water = keras.layers.BatchNormalization()(up_water)
    up_water = keras.layers.Reshape([image_size, image_size, -1])(up_water)
    concat_input = keras.layers.Concatenate()([up_water, input_image])
    patch_embedding = keras_cv.layers.PatchingAndEmbedding(project_dim=512, patch_size=8)(concat_input)
    vit_encode = vit_emmbed.MulTransformerBlock(patch_embedding, 512, 4, 768, 0)
    vit_mid = keras.layers.Dense(768, activation=train_config.activation)(vit_encode)
    vit_out = vit_mid[:, : -1, :]
    vit_reshape = keras.layers.Reshape([image_size, image_size, -1])(vit_out)
    vit_out = u_net_block.u_net(vit_reshape, times=4, activation=train_config.activation, drop=False)
    vit_reshape = keras.layers.Flatten()(vit_out)
    embed_image = LayerNom()([input_image, vit_reshape])
    return keras.models.Model(inputs=[input_image, input_watermark], outputs=[embed_image], name='embed_model')

def embed_model(psnr):
    """
    创建带 PSNR 的嵌入模型。

    参数:
    - psnr (float): 信噪比。

    返回:
    - keras.Model: 嵌入模型。
    """
    input_image = keras.layers.Input(shape=(image_size, image_size, 3))
    input_watermark = keras.layers.Input(shape=(water_size, water_size, 1))
    flatten_water = keras.layers.Flatten()(input_watermark)
    up_water = keras.layers.Dense(image_size ** 2, activation=train_config.activation)(flatten_water)
    up_water = keras.layers.BatchNormalization()(up_water)
    up_water = keras.layers.Reshape([image_size, image_size, -1])(up_water)
    concat_input = keras.layers.Concatenate()([up_water, input_image])
    patch_embedding = keras_cv.layers.PatchingAndEmbedding(project_dim=512, patch_size=8)(concat_input)
    vit_encode = vit_emmbed.MulTransformerBlock(patch_embedding, 512, 4, 768, 0)
    vit_mid = keras.layers.Dense(768, activation=train_config.activation)(vit_encode)
    vit_out = vit_mid[:, : -1, :]
    vit_reshape = keras.layers.Reshape([image_size, image_size, -1])(vit_out)
    vit_out = u_net_block.u_net(vit_reshape, times=4, activation=train_config.activation, drop=False)
    vit_reshape = keras.layers.Flatten()(vit_out)
    embed_image = LayerNormWithPsnr(psnr)([input_image, vit_reshape])
    return keras.models.Model(inputs=[input_image, input_watermark], outputs=[embed_image], name='embed_model')

class LayerNormWithPsnr(keras.layers.Layer):
    """
    带 PSNR 的层归一化。
    """
    def __init__(self, psnr, **kwargs):
        super().__init__(**kwargs)
        self.psnr = psnr
        self.var = self.cal_var()

    def cal_var(self):
        """
        计算方差。

        返回:
        - float: 计算得到的方差。
        """
        return math.sqrt(10 ** (-self.psnr / 10))

    def call(self, inputs, *args, **kwargs):
        """
        调用层归一化。

        参数:
        - inputs (tensor): 输入张量。

        返回:
        - tensor: 归一化后的张量。
        """
        image, add = inputs
        mean, var = tf.nn.moments(add, axes=[-1], keepdims=True)
        lay_add = ((add - mean) / tf.sqrt(var)) * self.var
        lay_add_reshape = tf.reshape(lay_add, tf.shape(image))
        return tf.clip_by_value(lay_add_reshape + image, 0.0, 1.0)

class LayerNorm(keras.layers.Layer):
    """
    层归一化。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        """
        调用层归一化。

        参数:
        - inputs (tensor): 输入张量。

        返回:
        - tensor: 归一化后的张量。
        """
        image, add = inputs
        mean, var = tf.nn.moments(add, axes=[-1], keepdims=True)
        lay_add = ((add - mean) / tf.sqrt(var)) * 0.01
        lay_add_reshape = tf.reshape(lay_add, tf.shape(image))
        return tf.clip_by_value(lay_add_reshape + image, 0.0, 1.0)
