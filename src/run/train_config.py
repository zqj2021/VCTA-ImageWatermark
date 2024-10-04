import time

# 激活函数
activation = 'gelu'

# 攻击方法
attack_method = 'all'

# 学习率设置
learning_rate = 0.0001
lr_stage = [50, 100, 150, 200, 300]
lr_decay_rate = [0.99, 0.995, 0.997, 0.999, 1.0]

# 批处理大小
batch_size = 96

# 每个周期的步数
epoch_per_steps = 10000 // batch_size

# 总训练周期数
epochs = 1000

# 峰值信噪比
PSNR = 40

# 模型名称
model_name = f"S128L64-F-{PSNR}"

# 主路径
main_path = './'

# 图像和水印的尺寸
image_size = 128
water_size = 8

# 日志目录，包含时间戳
time_str = time.strftime("%y-%m-%d.%H-%M-%S", time.localtime())
log_dir = f'./logs/{model_name}/{time_str}/'

# 嵌入模型目录
embed_model_dir = lambda :f'./models/{model_name}/{attack_method}/embed_model/'

# 提取模型目录
extr_model_dir = lambda :f'./models/{model_name}/{attack_method}/extr_model/'

# STN模型目录
stn_model_dir = lambda :f'./models/{model_name}/{attack_method}/stnfctxd/'

# 基础模型目录
base_embed_model_dir = f'./models/{model_name}/all/embed_model/'
base_extr_model_dir = f'./models/{model_name}/all/extr_model/'
base_stn_model_dir = f'./models/{model_name}/all/stnfctxd/'

# 图像保存目录
image_dir = f'./images/{model_name}/'

# 数据集基础路径
dataset_base_path = f'./datasets/'

# 训练和测试图像数据集名称
train_image = "ImageNet15K"
test_image = f'ImageNet4K'

# 训练和测试水印数据集名称
train_water = f'cifar50K'
test_water = f'cifar10K'
