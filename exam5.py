import ImageTools as it
import numpy as np


def ideal_low_filter(image, D0, save_path, show: bool = False):
    """
    频域平滑滤波器1: 理想低通滤波器
    """
    image = np.array(image.convert("L"))
    # 生成滤波器
    h, w = image.shape[:2]
    image_filter = np.ones((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            image_filter[i, j] = 0 if d > D0 else 1

    # 将滤波器应用到图像，生成理想低通滤波图像
    filtered_image = it.fromarray(
        it.restore255(filter_use(image, image_filter)))
    it.save_image(filtered_image, save_path)
    if show:
        it.compare_image_show(image, filtered_image)
    return filtered_image


def butterworth_low_filter(image, D0, rank, save_path, show: bool = False):
    """
    频域平滑滤波器2: Butterworth低通滤波器
    """
    image = np.array(image.convert("L"))
    # 生成滤波器
    h, w = image.shape[:2]
    image_filter = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            image_filter[i, j] = 1 / (1 + 0.414 * (d / D0) ** (2 * rank))

    # 将滤波器应用到图像，生成Butterworth低通滤波图像
    filtered_image = it.fromarray(
        it.restore255(filter_use(image, image_filter)))
    it.save_image(filtered_image, save_path)
    if show:
        it.compare_image_show(image, filtered_image)
    return filtered_image


def ideal_high_filter(image, D0, save_path, show: bool = False):
    """
    频域锐化滤波器1: 理想高通滤波器
    """
    image = np.array(image.convert("L"))
    # 生成滤波器
    h, w = image.shape[:2]
    image_filter = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            image_filter[i, j] = 0 if d < D0 else 1

    # 将滤波器应用到图像，生成Butterworth低通滤波图像
    filtered_image = it.fromarray(
        it.restore255(filter_use(image, image_filter)))
    it.save_image(filtered_image, save_path)
    if show:
        it.compare_image_show(image, filtered_image)
    return filtered_image


def butterworth_high_filter(image, D0, rank, save_path, show: bool = False):
    """
    频域锐化滤波器2: Butterworth高通滤波器
    """
    image = np.array(image.convert("L"))
    # 生成滤波器
    h, w = image.shape[:2]
    image_filter = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            image_filter[i, j] = 1 / (1 + (D0 / (d+1e-7)) ** (2 * rank))

    # 将滤波器应用到图像，生成Butterworth低通滤波图像
    filtered_image = it.fromarray(
        it.restore255(filter_use(image, image_filter)))
    it.save_image(filtered_image, save_path)
    if show:
        it.compare_image_show(image, filtered_image)
    return filtered_image


def filter_use(img, filter):
    """
    将图像与滤波器结合，生成对应的滤波图像数组
    """
    # 首先进行傅里叶变换
    f = np.fft.fft2(img)
    f_center = np.fft.fftshift(f)
    # 应用滤波器进行反变换
    S = np.multiply(f_center, filter)  # 频率相乘——l(u,v)*H(u,v)
    f_origin = np.fft.ifftshift(S)  # 将低频移动到原来的位置
    f_origin = np.fft.ifft2(f_origin)  # 使用ifft2进行傅里叶的逆变换
    f_origin = np.abs(f_origin)  # 设置区间
    return f_origin


if __name__ == "__main__":

    image = it.read_image('static/image_in/nana.jpg').convert("L")
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # ideal_low_filter(image, 50, save_path, show=True)
    # butterworth_low_filter(image, 50, 2, save_path, show=True)
    # ideal_high_filter(image, 50, save_path, show=True)
    butterworth_high_filter(image, 50, 2, save_path, show=True)
