from PIL import Image
import numpy as np
import ImageTools as it


def rgb2cmy(img, save_path, show: bool = False):
    # rgb转cmy
    r = np.array(img)[:, :, 0]
    g = np.array(img)[:, :, 1]
    b = np.array(img)[:, :, 2]
    c = 1-r
    m = 1-g
    y = 1-b
    result = Image.fromarray(np.dstack((c, m, y)))
    it.save_image(result, save_path)
    print('rgb2cmy成功')

    if show:
        it.compare_image_show(img, result)
    return result


def rgb2hsi(image, save_path, show: bool = False):

    image_array = np.array(image)

    # 将RGB值缩放到[0, 1]范围
    image_array = image_array / 255.0

    # 分割RGB通道
    r, g, b = np.split(image_array, 3, axis=2)

    # 色调（H）
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-8), -1, 1))
    h = np.where(b <= g, theta, 2 * np.pi - theta)
    h = h / (2 * np.pi)  # 将弧度转换为[0, 1]范围

    # 饱和度（S）
    s = 1 - 3 * np.minimum(r, np.minimum(g, b)) / (r + g + b + 1e-8)

    # 亮度（I）
    i = (r + g + b) / 3

    # 将HSI值缩放回[0, 255]范围
    h = h * 255
    s = s * 255
    i = i * 255

    # 创建新的HSI图像
    hsi_array = np.dstack((h, s, i)).astype(np.uint8)
    hsi_image = Image.fromarray(hsi_array)

    it.save_image(hsi_image, save_path)

    print('rgb2hsi成功')
    if show:
        it.compare_image_show(image, hsi_image)

    return hsi_image


def rgb2yuv(image, save_path, show: bool = False):
    rgb_image = np.array(image)
    # 定义RGB到YUV的转换矩阵
    r_coeff = 0.299
    g_coeff = 0.587
    b_coeff = 0.114
    # YUV 偏移量
    y_offset = 0
    u_offset = 128
    v_offset = 128
    # 转换矩阵
    conversion_matrix = np.array([
        [r_coeff, g_coeff, b_coeff],
        [-r_coeff / 1.772, -g_coeff / 1.772, (1 - b_coeff) / 1.402],
        [(1 - r_coeff) / 1.402, -g_coeff / 1.402, -b_coeff / 1.402]
    ])

    # 执行RGB到YUV转换
    reshaped_image = rgb_image.reshape(-1, 3)

    yuv_image = np.dot(reshaped_image, conversion_matrix.T)
    yuv_image[:, 0] += y_offset
    yuv_image[:, 1:] += u_offset, v_offset

    yuv_image = yuv_image.reshape(rgb_image.shape)
    yuv_image = np.uint8(np.clip(yuv_image, 0, 255))
    yuv_image = Image.fromarray(yuv_image)

    it.save_image(yuv_image, save_path)

    print('rgb2yuv成功')
    if show:
        it.compare_image_show(image, yuv_image)

    return yuv_image


def rgb2ycbcr(image, save_path, show: bool = False):
    rgb_array = np.array(image)

    red = rgb_array[:, :, 0]
    green = rgb_array[:, :, 1]
    blue = rgb_array[:, :, 2]

    # 将RGB值转换为YCbCr值
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.564 * (blue - y)
    cr = 0.713 * (red - y)

    # 将YCbCr值存储为NumPy数组
    ycbcr_array = np.zeros_like(rgb_array)
    ycbcr_array[:, :, 0] = y
    ycbcr_array[:, :, 1] = cb
    ycbcr_array[:, :, 2] = cr

    ycbcr_image = Image.fromarray(ycbcr_array.astype(np.uint8))

    it.save_image(ycbcr_image, save_path)

    print('rgb2ycbcr成功')
    if show:
        it.compare_image_show(image, ycbcr_image)

    return ycbcr_image


def com_color(image, save_path, show: bool = False):
    """
    RGB 补色
    """
    image_array = np.array(image)
    com_image_array = 255 - image_array
    com_image = Image.fromarray(com_image_array)

    it.save_image(com_image, save_path)
    print('RGB 补色成功')
    if show:
        it.compare_image_show(image, com_image)

    return com_image


def grayscale_transform(image, alpha, beta, save_path, show: bool = False):
    """
    灰度变换-线性
    > 你说为什么不做其他几种变换？拜托，这就是个课程作业而已......
    """
    gray_image = image.convert('L')
    image_array = np.array(gray_image)
    width, height = gray_image.size

    for i in range(height):
        for j in range(width):
            if alpha * image_array[i][j] + beta > 255:
                image_array[i][j] = 255
            else:
                image_array[i][j] = alpha * image_array[i][j] + beta

    gray_image = Image.fromarray(image_array)

    it.save_image(gray_image, save_path)
    print('线性灰度变换成功')
    if show:
        it.compare_image_show(image, gray_image)

    return gray_image


if __name__ == "__main__":
    image = it.read_image('static/image_in/nana.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # rgb2cmy(image, save_path, show=True)
    # rgb2hsi(image, save_path, show=True)
    # rgb2yuv(image, save_path, show=True)
    # rgb2ycbcr(image, save_path, show=True)
    # com_color(image, save_path, True)

    # grayscale_transform(image, 0.5, 5, save_path, True)
