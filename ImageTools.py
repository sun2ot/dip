from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def read_image(file_path):
    """
    读取图像文件并返回图像对象
    """
    try:
        image = Image.open(file_path)
        return image
    except Exception as e:
        print(f'图片读取出错：\n{e}')

def save_image(image, file_path):
    """
    保存图像对象到指定文件路径
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image.save(file_path)
    except Exception as e:
        print(f'图片保存出错：\n{e}')

def compare_image_show(origin, processed):
    # 显示原始图像和处理后的图像
    plt.subplot(121)
    plt.imshow(origin)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(122)
    plt.imshow(processed)
    plt.axis('off')
    plt.title('Processed')

    # 显示图像
    plt.show()

def algebraic_op_show(origin1, origin2, processed):
    # 显示进行代数运算的原始图像和处理后的图像
    plt.subplot(131)
    plt.imshow(origin1)
    plt.axis('off')
    plt.title('Original1')

    plt.subplot(132)
    plt.imshow(origin2)
    plt.axis('off')
    plt.title('Original2')

    plt.subplot(133)
    plt.imshow(processed)
    plt.axis('off')
    plt.title('Processed')

    # 显示图像
    plt.show()

def restore255(pixels):
    """
    将像素值缩放到 0-255 范围
    :param pixels: 图片的像素矩阵
    """
    pixels = ((pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels) + 0.1) * 255).astype(np.uint8)
    return pixels

def restore255_image(image):
    """
    将RGB图像的像素值归一化为0到255
    """
    input_array = np.array(image)
    channels = []
    for channel in range(3):
        result = restore255(input_array[:, :, channel])
        channels.append(result)
    norm_image = Image.fromarray(np.dstack(channels))
    return norm_image
