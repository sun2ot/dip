import numpy as np
import cv2
import ImageTools as it
from PIL import Image

input_image = Image.open('static/image_in/111.png')
input_image = input_image.convert('L')

# 读取图像
image = np.array(input_image)  # 这里假设你有一张灰度图像

# 计算直方图
histogram, bins = np.histogram(image.flatten(), 256, [0, 256])

# 计算累积分布函数
cdf = histogram.cumsum()
cdf_normalized = cdf * histogram.max() / cdf.max()

# 使用CDF重新映射像素值
equalized_image = np.interp(image, bins[:-1], cdf_normalized).astype(np.uint8)
equalized_image = Image.fromarray(equalized_image)

it.save_image(equalized_image, 'static/image_out/processed.png')


# 显示原图和均衡化后的图像
it.compare_image_show(image, equalized_image)

