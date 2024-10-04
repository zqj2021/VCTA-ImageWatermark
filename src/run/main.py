# 这是一个示例 Python 脚本。
import sys
import tensorflow as tf

# 检查是否有可用的 GPU 并设置内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)

import keras
import keras.optimizers
import load_dataset
import main_model_v2
import train_config
from my_callbacks import ShowImageCallback
from train_config import epochs, log_dir


def load_model():
    """
    加载嵌入和提取模型的权重，并编译主模型。

    返回:
    - keras.Model: 编译后的模型。
    """
    try:
        # raise OSError("Not loading the model")
        embed_model = main_model_v2.embed_model(train_config.PSNR)
        extr_model = main_model_v2.extr_model()
        embed_model.load_weights(train_config.base_embed_model_dir)
        extr_model.load_weights(train_config.base_extr_model_dir)
        print(f"Loaded model from {train_config.model_name}")
    except OSError:
        embed_model = main_model_v2.embed_model(train_config.PSNR)
        extr_model = main_model_v2.extr_model()
    model = main_model_v2.main_model(embed_model, extr_model)
    model.compile(keras.optimizers.AdamW(learning_rate=train_config.learning_rate))
    return model


def scheduler_lr(epoch, lr):
    """
    学习率调度器，根据训练的 epoch 调整学习率。

    参数:
    - epoch (int): 当前的训练周期。
    - lr (float): 当前的学习率。

    返回:
    - float: 调整后的学习率。
    """
    for e, l in zip(train_config.lr_stage, train_config.lr_decay_rate):
        if epoch < e:
            return lr * l
    return lr


# 主程序入口
if __name__ == '__main__':
    # 检查命令行参数以设置攻击方法
    if len(sys.argv) > 1:
        train_config.attack_method = sys.argv[1]
    print(f'attack : {train_config.attack_method}')

    # 加载训练和测试数据集
    train_dataset = load_dataset.ImageDataset()
    test_dataset = load_dataset.ImageDataset(_image=train_config.test_image, _water=train_config.test_water)

    # 设置回调函数
    callbacks = [
        ShowImageCallback(dataset=train_dataset),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10),
        tf.keras.callbacks.LearningRateScheduler(scheduler_lr)
    ]

    # 加载和编译模型
    _model = load_model()
    _model.summary()

    # 开始训练模型
    _model.fit(
        x=train_dataset.get_train_dataset(),
        epochs=epochs,
        steps_per_epoch=train_config.epoch_per_steps,
        callbacks=callbacks,
        validation_data=test_dataset.get_train_dataset(),
        validation_steps=20,
        initial_epoch=0
    )
