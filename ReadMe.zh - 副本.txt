# VCTA-ImageWatermark

本项目旨在实现一种嵌入和提取水印的算法，并对其进行各种攻击测试和性能评估。

## 目录结构

```
./
├── .git                         # Git版本控制文件夹
├── .idea                        # Pycharm 项目配置目录
├── datasets                     # 数据集
│   ├── COCO128                  # COCO128 数据集
│   └── Cifar128                 # Cifar128 数据集
├── logs                         # 日志文件
│   └── S128L64-F                # S128L64-F 日志
├── models                       # 保存的模型文件
│   └── S128L64-F                # S128L64-F 模型
├── plots                        # 图表和分析
│   └── MainPlot.ipynb           # 主绘图笔记代码
├── src                          # 源代码
│   ├── run                      # 训练相关代码
│   │   ├── attacks_layer.py     # 攻击层实现
│   │   ├── jpeg_attack.py       # JPEG 攻击实现
│   │   ├── load_dataset.py      # 数据集加载
│   │   ├── loss_and_metirc.py   # 损失和度量实现
│   │   ├── main.py              # 主程序入口
│   │   ├── main_model_v2.py     # 主模型实现
│   │   ├── my_callbacks.py      # 回调函数实现
│   │   ├── train_config.py      # 训练配置
│   │   ├── u_net_block.py       # U-Net 模块实现
│   │   └── vit_emmbed.py        # ViT 模块实现
│   └── test                     # 测试代码
│       ├── TestAttacks.py       # 攻击测试
│       ├── TestResults.py       # 结果测试
│       └── timeTest.py          # 时间测试
├── TestResult                   # 测试结果输出文件夹
├── ReadMe.pdf                   # ReadMe文档
└── ReadMe.md                    # ReadMe的Markdown文档
```

## 网络结构
### 嵌入网络结构
<div style="background-color: white; display: inline-block; padding: 10px;">
    <img src="Images/嵌入模型-M928.svg" alt="嵌入网络结构" style="max-width: 100%; height: auto;">
</div>

### 提取网络结构
<div style="background-color: white; display: inline-block; padding: 10px;">
    <img src="Images/提取模型M928.svg" alt="提取网络结构" style="max-width: 100%; height: auto;">
</div>



## 安装依赖

在开始运行项目之前，请确保安装了必要的依赖项。您可以使用以下命令安装所有依赖项：

```bash
pip install tensorflow==2.15.1
pip install keras-cv
pip install matplotlib
pip install opencv-python
```

### 环境需求
- 自TensorFlow 2.10.0 版本后, Tensorflow的GPU（CUDA）环境不再兼容Windows，请考虑使用[WSL (Windows Subsystem Linux)](https://learn.microsoft.com/zh-cn/windows/wsl/install) 或Linux系统运行该项目。
- TensorFlow == 2.15.0
- keras-cv
- matplotlib
- opencv-python

### 实验环境

- GPU: Tesla P40 (24G)
- CPU: E5 2680V2
- 内存: 64GB
- 操作系统：WSL


## 模型结构
`./src/run/main_model_v2.py` 包含了详细的嵌入如提取模型网络的结构定义。

### 主模型构建方法

```python
def main_model(embed, extr):
    pass
```

#### 功能

创建一个包含嵌入和提取功能、以及攻击层的完整模型用于训练。

#### 参数

- `embed (Model)`: 嵌入模型，负责将水印嵌入到图像中。
- `extr (Model)`: 提取模型，用于从可能经过攻击的图像中提取水印。

#### 返回值

- `keras.Model`: 组合了嵌入和提取功能的完整模型。


### 嵌入模型
```python
def embed_model():
    # input_image, input_watermark 使用函数式API构建模型输入
    # 嵌入逻辑实现
    # 使用U-Net与ViT模型等构建嵌入模型
    pass
```

#### 功能
构建水印嵌入模型。

#### 返回值

- `keras.Model`: 水印嵌入模型。


### 提取模型

```python
def extract_model(attacked_image):
    # 通过函数式API构建含水印图像输入
    # 提取逻辑实现
    # 包括使用U-Net与Vit结构等
    pass
```

#### 功能

构建水印提取模型

#### 返回值

- `keras.Model`: 水印提取模型。


## 使用方法

### 数据集加载

数据集加载由 `load_dataset.py` 文件处理。可以加载 COCO128 和 Cifar128 数据集。请将自定义的数据集放置在 `datasets` 文件夹中，目录结构如下所示：

```
datasets/
├── COCO128                  # COCO128 数据集
├── Cifar128                 # Cifar128 数据集
├── 自定义水印图片              # 水印图像数据集
├── 自定义载体图像              # 载体图像数据集
```
加载自定义数据集代码
```python
# 请确保 '自定义载体图像'与 '自定义水印图像'文件夹在./datasets文件夹中。
import load_dataset
test_dataset = load_dataset.ImageDataset(_image='自定义载体图像', _water='自定义水印图像')
```


### 模型训练和测试

`main_model_v2.py` 包含了嵌入模型和提取模型的实现。通过 `train_config.py` 配置训练参数。可以使用 `main.py` 作为训练模型的主入口。

### 训练过程可视化
`./src/run/my_callbacks.py`文件定义了训练过程中的回调参数，可以通过重写`on_epoch_end`等方法可视化每一轮的训练结果。

### `TensorBoard`使用方法
添加自定义的监视变量: 需要在自定义层中继承`keras.layers.Layer`，并在重写`forward`方法中使用`self.add_metric`添加监视到`TensorBoard`面板。

执行下列命令打开`TensorBoard`面板
```bash
tensorboard --logdir 'logs/模型名称/训练时间(yy-mm-dd.HH-MM-SS)/'
# https://localhost:6006
```

### 攻击测试

在 `TestAttacks.py` 文件中定义了各种攻击方法，如裁剪、Dropout、高斯噪声、椒盐噪声、高斯模糊等。可以通过调用这些方法对嵌入图像进行攻击测试。

### 结果测试

`TestResults.py` 文件用于对攻击后的图像进行提取水印并评估性能。结果将保存在 `TestResult` 目录中。

### 时间测试

`timeTest.py` 文件用于测试嵌入和提取水印的时间性能。

### 已保存模型参数加载
模型训练过程中，会自动保存平均BER < %10 且BER最低的EPOCH时模型在文件夹 `./models/模型名称/..` 中。
若要加载已保存模型用于进一步训练或测试，示例代码如下：
```python
import keras
import main_model_v2
import train_config
embed_model = main_model_v2.embed_model(train_config.PSNR)
extr_model = main_model_v2.extr_model()
embed_model.load_weights(train_config.base_embed_model_dir) #从嵌入模型路径中加载参数
extr_model.load_weights(train_config.base_extr_model_dir)   #从提取模型中加载参数
print(f"Loaded model from {train_config.model_name}")
```


## `train_config.py` 可配置参数

- `activation`: 激活函数，默认值为 'gelu'。
- `attack_method`: 攻击方法，默认值为 'all'。
- `learning_rate`: 初始学习率，默认值为 0.0001。
- `lr_stage`: 学习率调整的阶段。
- `lr_decay_rate`: 学习率衰减率。
- `batch_size`: 批处理大小，默认值为 96。
- `epoch_per_steps`: 每个周期的步数，根据 `batch_size` 计算。
- `epochs`: 总训练周期数，默认值为 1000。
- `PSNR`: 峰值信噪比，默认值为 40。
- `model_name`: 模型名称，根据 `PSNR` 动态生成。
- `main_path`: 主路径，默认值为 './'。
- `image_size`: 图像尺寸，默认值为 128。
- `water_size`: 水印尺寸，默认值为 8。
- `time_str`: 日志目录包含的时间戳，默认值为当前时间。
- `log_dir`: 日志目录，根据 `model_name` 和 `time_str` 动态生成。
- `embed_model_dir`: 嵌入模型目录，根据 `model_name` 动态生成。
- `extr_model_dir`: 提取模型目录，根据 `model_name` 动态生成。
- `stn_model_dir`: STN模型目录，根据 `model_name` 动态生成。
- `base_embed_model_dir`: 基础嵌入模型目录。
- `base_extr_model_dir`: 基础提取模型目录。
- `base_stn_model_dir`: 基础STN模型目录。
- `image_dir`: 图像保存目录。
- `dataset_base_path`: 数据集基础路径。
- `train_image`: 训练图像数据集名称。
- `test_image`: 测试图像数据集名称。
- `train_water`: 训练水印数据集名称。
- `test_water`: 测试水印数据集名称。

## 文件说明

- `attacks_layer.py`: 实现了各种攻击方法的类和函数。
- `jpeg_attack.py`: 实现了 JPEG 压缩攻击。
- `load_dataset.py`: 负责加载 COCO128 和 Cifar128 数据集。
- `loss_and_metirc.py`: 定义了损失函数和评估度量。
- `main.py`: 项目的主入口，用于训练和测试模型。
- `main_model_v2.py`: 包含嵌入模型和提取模型的实现。
- `my_callbacks.py`: 实现了回调函数，用于在训练过程中执行特定操作。
- `train_config.py`: 配置训练参数。
- `u_net_block.py`: 实现了 U-Net 模块。
- `vit_emmbed.py`: 实现了 ViT 嵌入。
- `TestAttacks.py`: 定义了攻击测试的方法。
- `TestResults.py`: 用于对攻击后的图像进行提取水印并评估性能。
- `timeTest.py`: 用于测试嵌入和提取水印的时间性能。

## 注意事项

1. 确保已安装 TensorFlow 和 Keras。
2. 使用 GPU 进行训练和测试，以提高性能，使用GPU进行训练时，确保显存 ≥ 16GB。
3. 结果文件会保存在 `TestResult` 目录中，请确保该目录存在并具有写权限。
4. 若运行过程中，项目目录存在问题，请确保根目录为项目目录。

## 联系方式

邮箱：1046714542@qq.com
