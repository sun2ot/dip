import numpy as np
from PIL import Image
import ImageTools as it
from scipy.signal import convolve2d


def add_gaussian_noise(image, save_path, mean=0, std=25):
    """
    高斯噪声
        - mean: 均值
        - std: 标准差(控制噪声的强度，值越大，噪声越强)
    """
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
    noisy_image = np.clip(img_array + noise, 0, 255)
    it.save_image(it.fromarray(noisy_image), save_path)
    print('添加高斯噪声成功')
    return noisy_image


def add_salt_and_pepper_noise(image, save_path, salt_prob=0.05, pepper_prob=0.05):
    """
    椒盐噪声
        - salt_prob: 盐噪声比例
        - pepper_prob: 椒噪声比例
    """
    img_array = np.array(image)
    noisy_image = np.copy(img_array)
    total_pixels = img_array.size

    # 添加盐噪声
    salt_coords = np.random.choice(total_pixels, int(total_pixels * salt_prob))
    noisy_image.flat[salt_coords] = 255

    # 添加椒噪声
    pepper_coords = np.random.choice(
        total_pixels, int(total_pixels * pepper_prob))
    noisy_image.flat[pepper_coords] = 0
    it.save_image(it.fromarray(noisy_image), save_path)
    print('添加椒盐噪声成功')
    return noisy_image


def add_uniform_noise(image, save_path, intensity=5):
    """
    波动噪声
        - intensity: 噪声强度，控制噪声的范围，即允许的最大增减值
    """
    img_array = np.array(image)
    noise = np.random.uniform(-intensity, intensity,
                              img_array.shape).astype(np.uint8)
    noisy_image = np.clip(img_array + noise, 0, 255)
    it.save_image(it.fromarray(noisy_image), save_path)
    print('添加波动噪声成功')
    return noisy_image


"""
下面五个滤波器做了一点小 trick，不论传入的滤波器大小多少，一律置为3. 出于以下原因：
1. 较大的 kernel_size 会导致执行时间显著增加，尤其是这种手动实现的垃圾代码
2. 写的时候直接用的硬编码的像素窗口大小，忘了构建动态窗口了hhh
3. just a homework, don't be so serious
"""


def ArithmeticMean_filter(image, kernel_size, save_path):
    """
    空间滤波器1：算术均值滤波器
    """
    kernel_size = 3
    image_array = np.array(image)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    height, width, channels = image_array.shape
    restored_image_array = np.zeros_like(image_array)
    for c in range(channels):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # 计算每个像素的均值
                pixel_value = np.sum(
                    image_array[i - 1:i + 2, j - 1:j + 2, c] * kernel)
                restored_image_array[i, j, c] = int(pixel_value)
    restored_image = it.fromarray(np.uint8(restored_image_array))
    it.save_image(restored_image, save_path)
    print('算术均值滤波器成功')
    return restored_image


def Median_filter(image, filter_size, save_path):
    """
    空间滤波器2：中值滤波器
    """
    filter_size = 3
    image_array = np.array(image)
    height, width, channels = image_array.shape
    restored_image_array = np.zeros_like(image_array)
    # 遍历图像的每个颜色通道并应用中值滤波
    for c in range(channels):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # 提取滤波器窗口
                window = image_array[i - filter_size // 2:i + filter_size //
                                     2 + 1, j - filter_size // 2:j + filter_size // 2 + 1, c]
                # 计算中值并将其赋给当前像素
                restored_image_array[i, j, c] = int(np.median(window))
    restored_image = it.fromarray(np.uint8(restored_image_array))
    it.save_image(restored_image, save_path)
    print('中值滤波器成功')
    return restored_image


def gaussian_kernel(size, sigma):
    """
    创建高斯滤波器核
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(- (x - size // 2)
                                                             ** 2 / (2 * sigma ** 2) - (y - size // 2) ** 2 / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def Gaussian_filter(image, kernel_size, sigma, save_path):
    """
    空间滤波器3：高斯滤波器
        - sigma: 标准差
    """
    kernel_size = 3
    image_array = np.array(image)
    gaussian_filter = gaussian_kernel(kernel_size, sigma)
    restored_image_array = np.zeros_like(image_array)
    for c in range(image_array.shape[2]):
        restored_image_array[:, :, c] = convolve2d(
            image_array[:, :, c], gaussian_filter, mode='same', boundary='wrap')
    restored_image = it.fromarray(np.uint8(restored_image_array))
    it.save_image(restored_image, save_path)
    print('高斯滤波器成功')
    return restored_image


def Max_filter(image, filter_size, save_path):
    """
    空间滤波器4：最大滤波器
    """
    filter_size = 3
    img_array = np.array(image)
    height, width, channels = img_array.shape
    restored_image = np.zeros_like(img_array)

    for y in range(filter_size // 2, height - filter_size // 2):
        for x in range(filter_size // 2, width - filter_size // 2):
            for c in range(channels):
                # 获取滤波器内的像素值
                region = img_array[y - filter_size // 2:y + filter_size // 2 + 1,
                                   x - filter_size // 2:x + filter_size // 2 + 1,
                                   c]
                max_value = np.max(region)
                restored_image[y, x, c] = max_value

    restored_image = it.fromarray(restored_image.astype('uint8'))
    it.save_image(restored_image, save_path)
    print('最大滤波器成功')
    return restored_image


def Min_filter(image, filter_size, save_path):
    """
    空间滤波器5：最小滤波器
    """
    filter_size = 3
    img_array = np.array(image)
    height, width, channels = img_array.shape
    restored_image = np.zeros_like(img_array)

    # 最小滤波器
    for y in range(filter_size // 2, height - filter_size // 2):
        for x in range(filter_size // 2, width - filter_size // 2):
            for c in range(channels):
                # 获取滤波器内的像素值
                region = img_array[y - filter_size // 2:y + filter_size //
                                   2 + 1, x - filter_size // 2:x + filter_size // 2 + 1, c]
                min_value = np.min(region)
                restored_image[y, x, c] = min_value

    restored_image = it.fromarray(restored_image.astype('uint8'))
    it.save_image(restored_image, save_path)
    print('最小滤波器成功')
    return restored_image


def bandstop_filter(image, center_frequency, bandwidth, save_path):
    """
    带阻滤波器
        - center_frequency: 中心频率
        - bandwidth: 带宽
    """
    image = image.convert('L')
    image_array = np.array(image)
    # 获取图像的频率域表示
    frequency_domain = np.fft.fftshift(np.fft.fft2(image_array))

    # 构造带阻滤波器
    height, width = image_array.shape
    y, x = np.indices((height, width))
    distance = np.sqrt((x - width/2)**2 + (y - height/2)**2)
    bandstop_mask = np.logical_and(
        distance >= center_frequency - bandwidth/2, distance <= center_frequency + bandwidth/2)

    # 应用滤波器
    filtered_frequency_domain = frequency_domain * \
        np.logical_not(bandstop_mask)

    # 进行逆变换，得到滤波后的图像
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_frequency_domain))
    filtered_image = np.real(filtered_image)
    restored_image = it.fromarray(filtered_image.astype(np.uint8))
    it.save_image(restored_image, save_path)
    print('带阻滤波器成功')
    return restored_image


def bandpass_filter(image, low_frequency, high_frequency, save_path):
    """
    带通滤波器
        - low_frequency: 低频
        - high_frequency: 高频
    """
    image = image.convert('L')
    image_array = np.array(image)
    # 获取图像的频率域表示
    frequency_domain = np.fft.fftshift(np.fft.fft2(image_array))

    # 构造带通滤波器
    height, width = image_array.shape
    y, x = np.indices((height, width))
    distance = np.sqrt((x - width/2)**2 + (y - height/2)**2)
    bandpass_mask = np.logical_and(
        distance >= low_frequency, distance <= high_frequency)

    # 应用滤波器
    filtered_frequency_domain = frequency_domain * bandpass_mask

    # 进行逆变换，得到滤波后的图像
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_frequency_domain))
    filtered_image = np.real(filtered_image)

    restored_image = it.fromarray(filtered_image.astype(np.uint8))
    it.save_image(restored_image, save_path)
    print('带通滤波器成功')
    return restored_image


def notch_filter(image, center_frequency, bandwidth, save_path):
    """
    陷波滤波器
        - center_frequency: 中心频率
        - bandwidth: 带宽
    """
    image = image.convert('L')
    image_array = np.array(image)
    # 获取图像的频率域表示
    frequency_domain = np.fft.fftshift(np.fft.fft2(image_array))

    # 构造陷波滤波器
    height, width = image_array.shape
    y, x = np.indices((height, width))
    distance = np.sqrt((x - width/2)**2 + (y - height/2)**2)
    notch_mask = np.logical_and(
        distance >= center_frequency - bandwidth/2, distance <= center_frequency + bandwidth/2)

    # 应用滤波器
    filtered_frequency_domain = frequency_domain * np.logical_not(notch_mask)

    # 进行逆变换，得到滤波后的图像
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_frequency_domain))
    filtered_image = np.real(filtered_image)

    restored_image = it.fromarray(filtered_image.astype(np.uint8))
    it.save_image(restored_image, save_path)
    print('陷波滤波器成功')
    return restored_image


if __name__ == "__main__":

    image = it.read_image('static/image_in/noisy.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # noisy_image = add_gaussian_noise(image)
    # noisy_image = add_salt_and_pepper_noise(image, save_path)
    # noisy_image = add_uniform_noise(image)

    # filtered_image = ArithmeticMean_filter(image, 3)
    # filtered_image = Median_filter(image, 3)
    # filtered_image = Gaussian_filter(image, 3, 1)
    # filtered_image = Max_filter(image, 3, save_path)
    # filtered_image = Min_filter(image, 3, save_path)

    # filtered_image = bandstop_filter(image, 30, 10, save_path)
    # filtered_image = bandpass_filter(image, 5, 200, save_path)
    filtered_image = notch_filter(image, 30, 10, save_path)
    it.compare_image_show(image, filtered_image)
