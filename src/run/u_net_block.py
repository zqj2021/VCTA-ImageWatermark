import keras.layers
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
import train_config


def conv_block(input_tensor, num_filters):
    """
    一个基本的卷积块，包含两个卷积层。

    参数:
    - input_tensor (tensor): 输入张量。
    - num_filters (int): 卷积核的数量。

    返回:
    - tensor: 卷积操作后的张量。
    """
    x = Conv2D(num_filters, (3, 3), activation=train_config.activation, padding='same')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation=train_config.activation, padding='same')(x)
    return x


def upsample_block(x, skip_features, num_filters):
    """
    上采样块，用于上采样和特征融合。

    参数:
    - x (tensor): 上采样输入张量。
    - skip_features (tensor): 跳跃连接的特征张量。
    - num_filters (int): 卷积核的数量。

    返回:
    - tensor: 上采样和特征融合后的张量。
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def u_net(x, times=2, filters=3, activation=train_config.activation, drop=True):
    """
    构建 U-Net 网络。

    参数:
    - x (tensor): 输入张量。
    - times (int): U-Net 的层数。
    - filters (int): 最后输出卷积核的数量。
    - activation (str): 激活函数。
    - drop (bool): 是否在上采样时使用 dropout。

    返回:
    - tensor: U-Net 的输出张量。
    """
    skip_connections = []
    # 下采样
    for _ in range(times):
        x = conv_block(x, 16 * 2 ** _)
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

    # 最底部
    x = conv_block(x, 16 * 2 ** times)

    # 上采样
    for _ in reversed(range(times)):
        if drop:
            x = keras.layers.SpatialDropout2D(rate=0.2)(x)
            skip_features = keras.layers.SpatialDropout2D(rate=0.2)(skip_connections[_])
        else:
            skip_features = skip_connections[_]
        x = upsample_block(x, skip_features, 16 * 2 ** _)

    # 最后的卷积层，确保输出通道数与输入相同
    outputs = Conv2D(filters, (1, 1), padding='same', activation=activation)(x)
    return outputs


def res_conv_block(x, num_filters, times=5):
    """
    构建一个带残差连接的卷积块。

    参数:
    - x (tensor): 输入张量。
    - num_filters (int): 卷积核的数量。
    - times (int): 残差连接的次数。

    返回:
    - tensor: 残差卷积块的输出张量。
    """
    for i in range(times):
        conv_out = conv_block(x, num_filters)
        x = keras.layers.Add()([conv_out, x])
    return x


def u_net_block(name, u_input, times=4, filters=3, activation='relu'):
    """
    创建一个 U-Net 模型块。

    参数:
    - name (str): 模型名称。
    - u_input (tensor): 输入张量。
    - times (int): U-Net 的层数。
    - filters (int): 最后输出卷积核的数量。
    - activation (str): 激活函数。

    返回:
    - Model: U-Net 模型块。
    """
    out = u_net(u_input, times, filters, activation)
    return keras.models.Model(inputs=u_input, outputs=out, name=name)(u_input)
