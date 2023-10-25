import ImageTools as it
import numpy as np
from PIL import Image


def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def manual_fft(image):
    data = np.array(image, dtype=np.complex128)
    height, width = data.shape

    # 计算填充尺寸
    padded_height = 2 ** int(np.ceil(np.log2(height)))
    padded_width = 2 ** int(np.ceil(np.log2(width)))

    # 执行填充
    padded_data = np.zeros((padded_height, padded_width), dtype=np.complex128)
    padded_data[:height, :width] = data

    # 执行行向量的FFT变换
    for i in range(padded_height):
        padded_data[i, :] = fft(padded_data[i, :])

    # 执行列向量的FFT变换
    for j in range(padded_width):
        padded_data[:, j] = fft(padded_data[:, j])

    return padded_data


def manual_ifft(data):
    height, width = data.shape

    # 执行行向量的逆FFT变换
    for i in range(height):
        data[i, :] = np.conjugate(fft(np.conjugate(data[i, :]))) / height

    # 执行列向量的逆FFT变换
    for j in range(width):
        data[:, j] = np.conjugate(fft(np.conjugate(data[:, j]))) / width

    return data


# 加载图像
image = Image.open('static/image_in/chong.png').convert('L')

# 将图像数据转换为NumPy数组
data = np.array(image)

# 执行FFT变换
fft_data = manual_fft(data)

# 执行逆FFT变换
ifft_data = manual_ifft(fft_data)

# 取实部并映射数值范围到0-255
ifft_data = np.real(ifft_data)
ifft_data = np.clip(ifft_data, 0, 255)
ifft_data = ifft_data.astype(np.uint8)

# 将数据转换回图像
output_image = Image.fromarray(ifft_data)


it.compare_image_show(image, output_image)
