import keras
import keras_cv.layers
import tensorflow as tf
from keras import layers
import train_config

def mlp(x, hidden_units, dropout_rate=0.0):
    """
    构建多层感知机（MLP）块。

    参数:
    - x (tensor): 输入张量。
    - hidden_units (list): 每一层的隐藏单元数。
    - dropout_rate (float): Dropout 概率。

    返回:
    - tensor: 经过 MLP 处理后的张量。
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=train_config.activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    """
    将图像划分为小块的层。
    """
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        """
        对图像应用分块操作。

        参数:
        - images (tensor): 输入图像。

        返回:
        - tensor: 分块后的图像。
        """
        _batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [_batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        """
        获取配置。

        返回:
        - dict: 配置信息。
        """
        config = super(Patches, self).get_config()
        config.update({
            "patch_size": self.patch_size
        })
        return config

class PatchEncoder(layers.Layer):
    """
    将图像块进行编码的层。
    """
    def __init__(self, target_size, patch_size, num_path, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.projection = layers.Dense(units=target_size)
        self.position_embed = layers.Embedding(input_dim=num_path,
                                               output_dim=target_size)
        self.patch = Patches(patch_size)
        position = tf.range(start=0, limit=num_path ** 2, delta=1)
        self.position = self.add_weight(name='position',
                                        shape=tf.shape(position),
                                        initializer=tf.initializers.Constant(position),
                                        trainable=False,
                                        dtype=tf.int32)
        self.target_size = target_size
        self.patch_size = patch_size
        self.num_path = num_path

    def call(self, inputs):
        """
        对图像块进行编码。

        参数:
        - inputs (tensor): 输入图像。

        返回:
        - tensor: 编码后的图像块。
        """
        position = self.position_embed(self.position)
        patch = self.patch(inputs)
        line_input = self.projection(patch)
        result = tf.add(line_input, position)
        return tf.keras.activations.relu(result)

    def get_config(self):
        """
        获取配置。

        返回:
        - dict: 配置信息。
        """
        config = super(PatchEncoder, self).get_config()
        config.update({
            'patch_size': self.patch_size,
            'num_path': self.num_path,
            'target_size': self.target_size
        })
        return config

def transform_block(embeddings, target_size, name_step):
    """
    构建 Transformer 块。

    参数:
    - embeddings (tensor): 输入嵌入张量。
    - target_size (int): 目标维度大小。
    - name_step (str): 块的名称。

    返回:
    - Model: Transformer 块模型。
    """
    norm_out = layers.LayerNormalization(epsilon=1e-6)(embeddings)
    attention_output = layers.MultiHeadAttention(
        num_heads=train_config.num_heads,
        key_dim=target_size,
        dropout=0.1
    )(norm_out, norm_out)
    add_out = layers.Add()([attention_output, embeddings])
    norm_out_2 = layers.LayerNormalization(epsilon=1e-6)(add_out)
    mlp_out = mlp(norm_out_2, hidden_units=[target_size, target_size * 2, target_size * 2, target_size],
                  dropout_rate=0.1)
    encode_patches = layers.Add()([mlp_out, add_out])
    return keras.models.Model(inputs=embeddings,
                              outputs=encode_patches,
                              name=f"Transform-Block-{name_step}")(embeddings)

def MulTransformerBlock(embed_image, project_dim, num_heads, mlp_dim, mlp_dropout=0.1, attention_dropout=0.1, depp=1, broad=3):
    """
    构建多 Transformer 块。

    参数:
    - embed_image (tensor): 输入嵌入图像张量。
    - project_dim (int): 投影维度。
    - num_heads (int): 注意力头数量。
    - mlp_dim (int): MLP 的维度。
    - mlp_dropout (float): MLP 的 Dropout 概率。
    - attention_dropout (float): 注意力层的 Dropout 概率。
    - depp (int): Transformer 块的深度。
    - broad (int): Transformer 块的宽度。

    返回:
    - tensor: 经过 Transformer 块处理后的张量。
    """
    out_trans = []
    for i in range(broad):
        x = embed_image
        for j in range(depp):
            x = keras_cv.layers.TransformerEncoder(project_dim, num_heads, mlp_dim, mlp_dropout, attention_dropout)(x)
        out_trans.append(x)
    return keras.layers.Concatenate()(out_trans)
