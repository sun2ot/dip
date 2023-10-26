import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import ImageTools as it


def ideal_lowpass_filter(image, cutoff_frequency, save_path, show: bool = False):
    """
    理想低通滤波器
    - cutoff_frequency: 截止频率
    """
    # 将图像转换为灰度图像
    image_gray = image.convert("L")

    # 将灰度图像转换为NumPy数组
    image_array = np.array(image_gray)

    # 获取图像的大小
    height, width = image_array.shape

    # 对图像进行傅里叶变换
    image_fft = fft2(image_array)

    # 创建一个理想低通滤波器
    filter = np.zeros((height, width))

    # 计算理想低通滤波器的频率响应
    for i in range(height):
        for j in range(width):
            distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
            if distance <= cutoff_frequency:
                filter[i, j] = 1

    # 在频域中应用滤波器
    filtered_image_fft = image_fft * filter

    # 计算滤波后图像的逆傅里叶变换
    filtered_image = np.real(ifft2(filtered_image_fft))

    # 切记图像归一化为0-255
    # plt.imshow()是自带归一化操作的，而保存到本地时没有该操作
    # 因此会导致显示的与保存下的图片看起来不一致
    filtered_image = it.fromarray(it.restore255(filtered_image)).convert('L')
    it.save_image(filtered_image, save_path)

    print("理想低通滤波器成功")
    if show:
        it.compare_image_show(image_gray, filtered_image)

    return filtered_image


def butterworth_lowpass_filter(image, cutoff_frequency, order, save_path, show: bool = False):
    """
    巴特沃斯低通滤波器
    - cutoff_frequency: 截止频率
    - order: 阶数
    """

    image_gray = image.convert("L")
    image_array = np.array(image_gray)
    height, width = image_array.shape

    # 对图像进行傅里叶变换
    image_fft = fft2(image_array)

    # 创建一个巴特沃斯低通滤波器
    filter = np.zeros((height, width))

    # 计算巴特沃斯低通滤波器的频率响应
    for i in range(height):
        for j in range(width):
            distance = np.sqrt((i - height/2)**2 + (j - width/2)**2)
            filter[i, j] = 1 / (1 + (distance / cutoff_frequency)**(2 * order))

    # 在频域中应用滤波器
    filtered_image_fft = image_fft * filter

    # 计算滤波后图像的逆傅里叶变换
    filtered_image = np.real(ifft2(filtered_image_fft))
    filtered_image = it.fromarray(it.restore255(filtered_image)).convert('L')
    it.save_image(filtered_image, save_path)

    print("巴特沃斯低通滤波器成功")
    if show:
        it.compare_image_show(image_gray, filtered_image)

    return filtered_image


def ideal_highpass_filter(image, D, save_path, show: bool = False):
    """
    高通滤波器
    - D: 截止频率
    """
    image = image.convert("L")
    image_array = np.array(image)

    # 进行傅里叶变换
    f_transform = fft2(image_array)

    # 创建高通滤波器
    rows, cols = image_array.shape
    center_row, center_col = rows // 2, cols // 2
    highpass_filter = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance > D:
                highpass_filter[i, j] = 1

    # 使用高通滤波器进行频域滤波
    filtered_image = ifft2(ifftshift(f_transform * highpass_filter)).real

    if show:
        it.compare_image_show(image, it.fromarray(filtered_image))

    # 将结果进行规范化
    filtered_image = it.fromarray(it.restore255(filtered_image)).convert('L')
    it.save_image(filtered_image, save_path)

    return filtered_image


def laplace_filter(image, save_path, show: bool = False):
    """
    拉普拉斯滤波器
    """
    image = image.convert("L")
    image_array = np.array(image)

    # 进行傅里叶变换
    f_transform = fft2(image_array)

    # 创建拉普拉斯滤波器
    rows, cols = image_array.shape
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # 扩展拉普拉斯滤波器以匹配图像尺寸
    filter_rows, filter_cols = laplacian_filter.shape
    padded_filter = np.zeros((rows, cols))
    padded_filter[:filter_rows, :filter_cols] = laplacian_filter

    # 使用拉普拉斯滤波器进行频域滤波
    filtered_image = ifft2(ifftshift(f_transform * padded_filter)).real

    # 将结果进行规范化
    filtered_image = it.fromarray(it.restore255(filtered_image)).convert('L')
    it.save_image(filtered_image, save_path)
    if show:
        it.compare_image_show(image, filtered_image)
    return filtered_image


if __name__ == "__main__":
    # 打开图像
    image = it.read_image('static/image_in/eject.png')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.png'

    ideal_lowpass_filter(image, 220, save_path, True)
    # butterworth_lowpass_filter(image, 50, 1, save_path, True)

    # ideal_highpass_filter(image, 10, save_path, True)
    # laplace_filter(image, save_path, True)
