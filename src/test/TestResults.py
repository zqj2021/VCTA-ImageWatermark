# %%
import imageio
import keras.models

from src.run import load_dataset, main_model_v2
from src.run.attacks_layer import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)
import os
import shutil
from TestAttacks import *

# 加载数据集
datasets = load_dataset.ImageDataset("COCO128", "Cifar130", shuffle=False, batch_size2=128, image_size2=256)
images = list(iter(datasets.images1))
waters = list(iter(datasets.waters_binary1))

class WatermarkAttack:
    """
    水印攻击类，用于定义不同类型的攻击。
    """
    def __init__(self, attack_method, attack_function, *attack_parameters):
        self.attack_method = attack_method
        self.attack_function = attack_function
        self.attack_parameters = attack_parameters

# 定义各种攻击方法
cropping = WatermarkAttack('裁剪', random_cutout, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
dropout = WatermarkAttack('Dropout', random_dropout, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
gaussian_noise = WatermarkAttack('高斯噪声', random_gaussnoise, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2)
salt_and_pepper_noise = WatermarkAttack('椒盐噪声', random_papersalt, 0.01, 0.02, 0.03, 0.04, 0.05)
gaussian_blur = WatermarkAttack('高斯模糊', random_gaussglur, 0.0001, 0.5, 1, 2)
median_filter = WatermarkAttack('中值滤波', random_mediaGlur, 1, 3, 5, 7)
jpeg_compression = WatermarkAttack('JPEG压缩', random_jpeg, 30, 40, 50, 60, 70, 80, 90)
jpeg_compression2 = WatermarkAttack('模拟JPEG压缩', jpeg_attack.jpegQ, 30, 40, 50, 60, 70, 80, 90)
all_attacks = [cropping, dropout, gaussian_noise, gaussian_blur, salt_and_pepper_noise, jpeg_compression, median_filter, jpeg_compression2]

def binarize_images(images, threshold=0.5):
    """
    根据给定的阈值将图像二值化。假设图像像素值在 [0, 1] 范围内。

    参数:
    - images (numpy.ndarray): 要二值化的图像数组。
    - threshold (float): 二值化阈值。

    返回:
    - numpy.ndarray: 二值化后的图像。
    """
    return (images > threshold).astype(np.uint8)

def calculate_ber(input_images, reference_images, threshold=0.5):
    """
    计算每个图像的输入图像和参考图像之间的比特错误率（BER）。

    参数:
    - input_images: 输入图像的 numpy 数组，形状为 (N, H, W, 1)。
    - reference_images: 参考图像的 numpy 数组，形状为 (N, H, W, 1)。
    - threshold: 二值化阈值。

    返回:
    - numpy.ndarray: 形状为 (N,) 的 BER 计算结果数组。
    """
    binarized_input = (input_images >= threshold).astype(np.int8)
    binarized_reference = (reference_images >= threshold).astype(np.int8)
    bers = np.zeros(input_images.shape[0], np.float32)
    for i in range(input_images.shape[0]):
        errors = np.sum(binarized_input[i] != binarized_reference[i])
        total_bits = np.prod(input_images[i].shape)
        bers[i] = errors / total_bits
    return bers

def save_image(filename: str, image):
    """
    保存图像到指定文件。

    参数:
    - filename (str): 文件名。
    - image: 要保存的图像。
    """
    image = np.squeeze(image)
    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True).numpy()
    imageio.imwrite(filename, image)

def delete_files_in_folder(folder_path):
    """
    删除文件夹中的所有文件。

    参数:
    - folder_path (str): 文件夹路径。
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    shutil.rmtree(folder_path)

def expand_pixels(input_array, n):
    """
    扩展数组的每个像素。

    参数:
    - input_array (numpy.ndarray): 输入数组。
    - n (int): 扩展倍数。

    返回:
    - numpy.ndarray: 扩展后的数组。
    """
    expanded_y = np.repeat(input_array, n, axis=2)
    expanded_x = np.repeat(expanded_y, n, axis=3)
    return expanded_x

# %%
class AttackRecord:
    """
    攻击记录类，用于存储攻击结果。
    """
    def __init__(self):
        super().__init__()
        self.img_index = None
        self.psnr = None
        self.ssim = None
        self.ber = None
        self.attack_method = None
        self.attack_arg = None

tempX = None

class BatchAttack:
    """
    批量攻击类，用于对一批图像进行攻击并记录结果。
    """
    def __init__(self, embed_model, extr_model, batch_image, batch_water, attack, image_indexs):
        super().__init__()
        self.bers = None
        self.extr_waters = None
        self.attacks_image = None
        self.att_arg = None
        self.ssims = None
        self.psnrs = None
        self.embed_image = None
        self.embed_model = embed_model
        self.extr_model = extr_model
        self.image = batch_image.numpy()
        self.water = batch_water.numpy()
        self.attack: WatermarkAttack = attack
        self.image_index = image_indexs

    def start_attack(self):
        """
        开始对一批图像进行攻击。
        """
        patch_image = jpeg_attack.patchs(self.image, image_size=256)
        image_list = tf.split(patch_image, 4, axis=1)
        embeds = []
        psnrs = []
        ssims = []
        attackImages = []
        extrWaters = []
        berS = []
        for image in image_list:
            image = tf.squeeze(image, axis=1)
            embed_image = self.embed_model.predict([image, self.water])
            psnr = tf.image.psnr(image, embed_image, 1.0).numpy()
            ssim = tf.image.ssim(image, embed_image, 1.0).numpy()
            self.att_arg = self.attack.attack_parameters
            attacks_image = []
            extr_waters = []
            bers = []
            for arg in self.attack.attack_parameters:
                att_image = self.attack.attack_function(embed_image, arg)
                attacks_image.append(att_image)
                extr_water = self.extr_model.predict(att_image)
                extr_waters.append(extr_water)
                ber = calculate_ber(self.water, extr_water)
                bers.append(ber)
            embeds.append(embed_image)
            psnrs.append(psnr)
            ssims.append(ssim)
            attackImages.append(attacks_image)
            extrWaters.append(extr_waters)
            berS.append(bers)

        self.ssims = np.mean(np.array(ssims), axis=0)
        self.psnrs = np.mean(np.array(psnrs), axis=0)
        embed_image_m = tf.transpose(np.array(embeds), perm=[1, 0, 2, 3, 4])
        re_image = jpeg_attack.repatchs(embed_image_m, image_size=256, c=3).numpy()
        self.embed_image = re_image
        self.attacks_image = []
        for arg in self.attack.attack_parameters:
            att_image = self.attack.attack_function(self.embed_image, arg)
            self.attacks_image.append(att_image)
        tempwater = np.mean(np.array(extrWaters), axis=0)
        self.extr_waters = expand_pixels(tempwater, 32)
        self.bers = np.mean(np.array(berS), axis=0)

    def save_images(self):
        """
        保存攻击后的图像和水印图像。
        """
        start_index = self.image_index
        batch_size = len(self.image)
        for i in range(batch_size):
            save_image(
                f"TestResult/{train_config.model_name}/{self.attack.attack_method}/含水印图像/{start_index + i}_psnr_{self.psnrs[i]}.bmp",
                self.embed_image[i])
            save_image(
                f"TestResult/{train_config.model_name}/{self.attack.attack_method}/原始图像/{start_index + i}.bmp",
                self.image[i])
            save_image(
                f"TestResult/{train_config.model_name}/{self.attack.attack_method}/水印图像/{start_index + i}.bmp",
                self.water[i])
            for j in range(len(self.att_arg)):
                print(self.bers[j][i])
                save_image(
                    f"TestResult/{train_config.model_name}/{self.attack.attack_method}/攻击后图像/{start_index + i}_{self.attack.attack_method}_{self.att_arg[j]}.bmp",
                    self.attacks_image[j][i])
                save_image(
                    f"TestResult/{train_config.model_name}/{self.attack.attack_method}/提取水印图像/{start_index + i}_{self.att_arg[j]}_ber_{self.bers[j][i]}.bmp",
                    self.extr_waters[j][i])

    def append_all(self, records):
        """
        将所有攻击结果追加到记录中。

        参数:
        - records (list): 攻击记录列表。
        """
        for i in range(len(self.image)):
            for j in range(len(self.att_arg)):
                rec = AttackRecord()
                rec.img_index = self.image_index + i
                rec.attack_arg = self.att_arg[j]
                rec.ssim = self.ssims[i]
                rec.psnr = self.psnrs[i]
                rec.ber = self.bers[j][i]
                rec.attack_method = self.attack.attack_method
                records.append(rec)

def attack(att: WatermarkAttack, embed_model, extr_model, save_files):
    """
    对图像进行攻击并记录结果。

    参数:
    - att (WatermarkAttack): 水印攻击对象。
    - embed_model (Model): 嵌入模型。
    - extr_model (Model): 提取模型。
    - save_files (bool): 是否保存攻击后的文件。

    返回:
    - list: 攻击记录列表。
    """
    res_record = []
    try:
        os.mkdir(f"TestResult/{train_config.model_name}/")
    except FileExistsError:
        pass
    if save_files:
        try:
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/原始图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/含水印图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/攻击后图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/提取水印图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/水印图像")
        except FileExistsError:
            delete_files_in_folder(f"TestResult/{train_config.model_name}/{att.attack_method}/")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/原始图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/含水印图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/攻击后图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/提取水印图像")
            os.mkdir(f"TestResult/{train_config.model_name}/{att.attack_method}/水印图像")

    start_index = 0
    for bat_img, bat_water in zip(images, waters):
        batch_attack = BatchAttack(embed_model, extr_model, bat_img, bat_water, att, start_index)
        batch_attack.start_attack()
        if save_files:
            batch_attack.save_images()
        batch_attack.append_all(res_record)
        start_index += len(bat_img)
    return res_record

# 设置 PSNR 值并加载模型
all_psnrs = [50]
for e_psnr in all_psnrs:
    embed_model = main_model_v2.embed_model(e_psnr)
    extr_model = main_model_v2.extr_model()
    embed_model.load_weights(train_config.embed_model_dir())
    extr_model.load_weights(train_config.extr_model_dir())

    for att in all_attacks:
        res = attack(att, embed_model, extr_model, True)
        psnrs = list(map(lambda x: x.psnr, res))
        psnr = np.average(psnrs)
        ssims = list(map(lambda x: x.ssim, res))
        ssim = np.average(ssims)
        write_string1 = f"PSNR:{'%.4f' % psnr}\tSSIM:{'%.4f' % ssim}\nimage\tPSNR\t攻击方式\t攻击参数\tBER\n"
        write_string2 = "攻击方式\t攻击参数\t平均BER\n"
        for arg in att.attack_parameters:
            rec = list(filter(lambda x: x.attack_arg == arg, res))
            for r in rec:
                r: AttackRecord
                write_string1 += f"{r.img_index}\t{'%.4f' % r.psnr}\t{r.attack_method}\t{r.attack_arg}\t{'%.4f' % r.ber}\n"
            bers = list(map(lambda x: x.ber, rec))
            average = np.average(bers)
            write_string2 += f"{att.attack_method}\t{arg}\t{'%.4f' % average}\n"

        with open(f"TestResult/{train_config.model_name}/{att.attack_method}_PSNR_{e_psnr}_result.txt", 'w') as f:
            text_res = write_string2 + "\n" + write_string1
            f.write(text_res)
            f.flush()
        res.clear()

# %% 输出去所有中间结果
sample_image = images[0][:1]
sample_water = waters[0][11:12]

def get_layer_outputs(model, sample):
    """
    获取模型每一层的输出。

    参数:
    - model (Model): 输入模型。
    - sample (tensor): 输入样本。

    返回:
    - dict: 每一层的名称和对应的激活。
    """
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(sample)
    layer_activations = {}
    for layer, activation in zip(model.layers, activations):
        layer_activations[layer.name] = activation
    return layer_activations

# 获取所有层的输出并显示特定层的输出
all_outs_layer = get_layer_outputs(embed_model, [sample_image, sample_water])
unet_out = all_outs_layer['conv2d_19']
import matplotlib.pyplot as plt

plt.imshow(unet_out[0] * 10)
plt.show()
# %%
import numpy as np

def expand_pixels(input_array, n):
    """
    扩展数组的每个像素。

    参数:
    - input_array (numpy.ndarray): 输入数组。
    - n (int): 扩展倍数。

    返回:
    - numpy.ndarray: 扩展后的数组。
    """
    expanded_y = np.repeat(input_array, n, axis=0)
    expanded_x = np.repeat(expanded_y, n, axis=1)
    return expanded_x

# 创建一个简单的示例数组
array = np.array([[1, 2],
                  [3, 4]])

# 调用函数，每个像素扩展为3x3块
expanded_array = expand_pixels(array, 3)

print("Original Array:\n", array)
print("Expanded Array:\n", expanded_array)
