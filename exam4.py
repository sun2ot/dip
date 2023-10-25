import numpy as np
from PIL import Image
import ImageTools as it


def discrete_fourier_transform(image):
    # 将图像转换为灰度
    image = image.convert("L")

    # 获取图像尺寸
    width, height = image.size

    # 初始化傅里叶变换结果矩阵
    f_transform = np.zeros((height, width), dtype=complex)

    for u in range(width):
        for v in range(height):
            f_uv = 0
            for x in range(width):
                for y in range(height):
                    pixel_value = image.getpixel((x, y))
                    f_uv += pixel_value * \
                        np.exp(-2j * np.pi * (u * x / width + v * y / height))
            f_transform[v, u] = f_uv

    return f_transform


def inverse_discrete_fourier_transform(f_transform):
    height, width = f_transform.shape
    reconstructed_image = np.zeros((height, width), dtype=float)

    for x in range(width):
        for y in range(height):
            pixel_value = 0
            for u in range(width):
                for v in range(height):
                    f_uv = f_transform[v, u]
                    pixel_value += f_uv * \
                        np.exp(2j * np.pi * (u * x / width + v * y / height))
            reconstructed_image[y, x] = abs(pixel_value) / (width * height)

    return reconstructed_image


def fourier_transform(image, save_path, show: bool = False):
    # 执行离散傅里叶变换
    f_transform = discrete_fourier_transform(image)

    # 执行逆离散傅里叶变换
    reconstructed_image = inverse_discrete_fourier_transform(f_transform)

    reconstructed_image = it.fromarray(it.restore255(reconstructed_image))
    it.save_image(reconstructed_image, save_path)
    print('傅里叶变换成功')
    if show:
        it.compare_image_show(image.convert("L"), reconstructed_image)

    return reconstructed_image


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


def fast_fourier_transform(image, save_path, show: bool = False):
    image = image.convert('L')

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

    it.save_image(output_image, save_path)
    if show:
        it.compare_image_show(image, output_image)


if __name__ == "__main__":
    # 打开图像
    image = it.read_image('static/image_in/chong.png')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # fourier_transform(image, save_path, True)
    fast_fourier_transform(image, save_path, True)
