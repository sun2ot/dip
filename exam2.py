from PIL import Image
import numpy as np
import ImageTools as it


def point_op_reverse(input_image, save_path: str, show: bool = False):
    """
    点运算1-反转变换
    :param input_image: 待处理图片
    :param save_path: 处理后图片路径
    :param show: 是否显示处理前后对比图
    :return: void
    """

    # 获取图像的宽度和高度
    width, height = input_image.size

    # 创建一个新的图像对象，用于存储反转后的像素数据
    reversed_image = Image.new('RGB', (width, height))

    # 遍历每个像素点，进行反转变换
    for x in range(width):
        for y in range(height):
            r, g, b = input_image.getpixel((x, y))
            reversed_r = 255 - r
            reversed_g = 255 - g
            reversed_b = 255 - b
            reversed_image.putpixel((x, y), (reversed_r, reversed_g, reversed_b))

    it.save_image(reversed_image, save_path)
    print('反转成功')

    # 展示对比图
    if show:
        it.compare_image_show(input_image, reversed_image)


def point_op_log(input_image, save_path: str, c: float = 1.0, show: bool = False):
    """
    点运算2-对数变换
    """
    # 使用 NumPy 数组进行对数变换
    input_array = np.array(input_image)

    # rgb 三通道
    log_transformed_r = c * np.log(np.where(input_array[:, :, 0] < 255, input_array[:, :, 0] + 1, 1))  # 红色通道
    log_transformed_g = c * np.log(np.where(input_array[:, :, 1] < 255, input_array[:, :, 1] + 1, 1))  # 绿色通道
    log_transformed_b = c * np.log(np.where(input_array[:, :, 2] < 255, input_array[:, :, 2] + 1, 1))  # 蓝色通道

    # 将像素值缩放到 0-255 范围（必选） ，否则是黑图，因为都是小像素值
    log_transformed_r = it.restore255(log_transformed_r)
    log_transformed_g = it.restore255(log_transformed_g)
    log_transformed_b = it.restore255(log_transformed_b)

    # 合并通道并创建对数变换后的 RGB 图像
    # np.dstack 沿着 deep(第三维度) 堆叠
    log_transformed_image = Image.fromarray(np.dstack((log_transformed_r, log_transformed_g, log_transformed_b)))
    it.save_image(log_transformed_image, save_path)
    print('对数变换成功')

    # 展示对比图
    if show:
        it.compare_image_show(input_image, log_transformed_image)


def point_op_powertrans(input_image, save_path: str, c: float = 1.0, gamma: float = 1.0, show: bool = False):
    """
    点运算3-幂次变换
    :param gamma: 大于1变暗，小于1变亮
    """
    input_array = np.array(input_image)
    channels = []  # 变换后的三通道数组
    for channel in range(3):
        # 执行幂次变换(先变再缩)
        result = it.restore255(c * np.power(input_array[:, :, channel], gamma))
        channels.append(result)
    power_trans_image = Image.fromarray(np.dstack(channels))
    it.save_image(power_trans_image, save_path)
    print('幂次变换成功')
    # 展示对比图
    if show:
        it.compare_image_show(input_image, power_trans_image)


def point_op_contrast_stretching(input_image, save_path: str, min_pixel: int, max_pixel: int, show: bool = False):
    """
    点运算4-对比度拉伸
    :param min_pixel: 拉伸的最小像素
    :param max_pixel:  拉伸的最大像素
    """
    input_array = np.array(input_image)
    channels = []  # 变换后的三通道数组
    for channel in range(3):
        # 执行对比度拉伸
        result = np.interp(input_array[:, :, channel], (0, 255), (min_pixel, max_pixel)).astype(np.uint8)
        channels.append(result)
    contrast_stretching_image = Image.fromarray(np.dstack(channels))
    it.save_image(contrast_stretching_image, save_path)
    print('对比度拉伸成功')
    # 展示对比图
    if show:
        it.compare_image_show(input_image, contrast_stretching_image)


def point_op_gray_slice(input_image, save_path: str, min_pixel: int, max_pixel: int, show: bool = False):
    """
    点运算5-灰度级切片
    :param min_pixel: 切片最小像素
    :param max_pixel: 切片最大像素
    """
    gray_image = input_image.convert('L')
    sliced_image = Image.new('L', gray_image.size)

    width, height = gray_image.size
    for i in range(width):
        for j in range(height):
            pixel = gray_image.getpixel((i, j))
            if min_pixel <= pixel <= max_pixel:
                # 阈值范围内保持原像素值
                sliced_image.putpixel((i, j), pixel)
            else:
                # 超出阈值范围的像素值设为0
                sliced_image.putpixel((i, j), 0)

    it.save_image(sliced_image, save_path)
    print('灰度级切片成功')
    # 展示对比图
    if show:
        it.compare_image_show(input_image, sliced_image)


def point_op_bitplane_slice(input_image, save_path: str, bit: int, show: bool = False):
    """
    点运算6-位平面切片
    :param bit: 第几位平面(0-7)
    """
    if bit not in range(0, 8):
        raise Exception('bit值应该为0到7的整数!')

    gray_image = input_image.convert('L')
    bit_plane = Image.new('L', gray_image.size)

    width, height = gray_image.size
    for i in range(width):
        for j in range(height):
            pixel = gray_image.getpixel((i, j))
            bit_value = (pixel >> bit) & 1  # 提取特定位的值
            new_pixel = bit_value * 255
            bit_plane.putpixel((i, j), new_pixel)

    it.save_image(bit_plane, save_path)
    print('位平面切片成功')
    # 展示对比图
    if show:
        it.compare_image_show(input_image, bit_plane)


# -----------------------------------------------------------

def image_cal1(image1, image2, save_path: str, op: str, show: bool = False):
    """
    代数运算：加减乘除
    :param op: add | subtract | multiply | divide
    """
    if image1.size != image2.size:
        raise Exception('两张图片的大小不一致!')

    # 检查 operation 参数是否合法
    if op not in ('add', 'subtract', 'multiply', 'divide'):
        raise ValueError("非法操作! 仅支持 'add', 'subtract', 'multiply' and 'divide'")

    result_image = Image.new('RGB', image1.size)
    new_pixel = np.zeros(result_image.size)

    for x in range(image1.width):
        for y in range(image1.height):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            if op == 'add':
                new_pixel = (pixel1[0] + pixel2[0],
                             pixel1[1] + pixel2[1],
                             pixel1[2] + pixel2[2])
            elif op == 'subtract':
                new_pixel = (pixel1[0] - pixel2[0],
                             pixel1[1] - pixel2[1],
                             pixel1[2] - pixel2[2])
            elif op == 'multiply':
                new_pixel = (pixel1[0] * pixel2[0],
                             pixel1[1] * pixel2[1],
                             pixel1[2] * pixel2[2])
            elif op == 'divide':
                # 确保不会除以0
                new_pixel = (pixel1[0] // max(pixel2[0], 1),
                             pixel1[1] // max(pixel2[1], 1),
                             pixel1[2] // max(pixel2[2], 1))

            result_image.putpixel((x, y), new_pixel)
    # 归一化为0-255
    result_image = it.restore255_image(result_image)
    it.save_image(result_image, save_path)
    print('加减乘除成功')
    # 展示对比图
    if show:
        it.algebraic_op_show(image1, image2, result_image)


def image_cal2(image, save_path: str, show: bool = False):
    """
    代数运算：非运算
    """
    input_array = np.array(image)
    print(input_array.shape)
    channels = []
    for channel in range(3):
        result = 255 - input_array[:, :, channel]
        channels.append(result)
    not_image = Image.fromarray(np.dstack(channels))
    it.save_image(not_image, save_path)
    print('非运算成功')
    # 展示对比图
    if show:
        it.compare_image_show(image, not_image)


def image_cal3(image1, image2, save_path: str, op: str, show: bool = False):
    """
    与、或运算
    """
    # 确保两个图像具有相同的大小
    if image1.size != image2.size:
        raise ValueError("图像大小不一致")

    # 检查 operation 参数是否合法
    if op not in ('&', '|', '^'):
        raise ValueError("非法操作! 仅支持 '&', '|' and '^'")

    result_image = Image.new('RGB', image1.size)

    for x in range(image1.width):
        for y in range(image1.height):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            # 执行运算
            if op == '&':
                result_pixel = tuple(p1 & p2 for p1, p2 in zip(pixel1, pixel2))
            elif op == '|':
                result_pixel = tuple(p1 | p2 for p1, p2 in zip(pixel1, pixel2))
            elif op == '^':
                result_pixel = tuple(p1 ^ p2 for p1, p2 in zip(pixel1, pixel2))

            result_image.putpixel((x, y), result_pixel)

    it.save_image(result_image, save_path)
    print('与/或/异或 运算成功')
    if show:
        it.algebraic_op_show(image1, image2, result_image)


def histogram_eq(input_image, save_path: str, show: bool = False):

    input_image = input_image.convert('L')

    image = np.array(input_image)  # 这里假设你有一张灰度图像

    # 计算直方图
    histogram, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累积分布函数
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # 使用CDF重新映射像素值
    equalized_image = np.interp(image, bins[:-1], cdf_normalized).astype(np.uint8)
    equalized_image = Image.fromarray(equalized_image)

    it.save_image(equalized_image, save_path)
    print('直方图均衡化成功')
    if show:
        it.compare_image_show(image, equalized_image)


def linear_smoothing_filter(image, kernel_size, save_path: str, show: bool = False):
    """
    线性平滑滤波器
    :param kernel_size: 滤波器大小(太大会很慢很慢,推荐3-5)
    """
    width, height = image.size
    border = kernel_size // 2

    smoothed_image = Image.new('RGB', (width, height))

    for x in range(border, width - border):
        for y in range(border, height - border):
            region_r, region_g, region_b = [], [], []
            for i in range(-border, border + 1):
                for j in range(-border, border + 1):
                    pixel_r, pixel_g, pixel_b = image.getpixel((x + i, y + j))
                    region_r.append(pixel_r)
                    region_g.append(pixel_g)
                    region_b.append(pixel_b)
            average_r = sum(region_r) // len(region_r)
            average_g = sum(region_g) // len(region_g)
            average_b = sum(region_b) // len(region_b)
            smoothed_image.putpixel((x, y), (average_r, average_g, average_b))
    it.save_image(smoothed_image, save_path)
    print('线性平滑滤波器成功')
    if show:
        it.compare_image_show(image, smoothed_image)


def middle_smoothing_filter(image, kernel_size, save_path: str, show: bool = False):
    """
    中值滤波器成功
    """
    width, height = image.size
    smoothed_image = Image.new('RGB', (width, height))
    filter_radius = kernel_size // 2

    # 遍历图像的每个像素
    for x in range(width):
        for y in range(height):
            red_values, green_values, blue_values = [], [], []

            # 遍历滤波器窗口内的像素
            for i in range(-filter_radius, filter_radius + 1):
                for j in range(-filter_radius, filter_radius + 1):
                    pixel_x = x + i
                    pixel_y = y + j

                    # 检查像素是否在图像范围内
                    if 0 <= pixel_x < width and 0 <= pixel_y < height:
                        pixel = image.getpixel((pixel_x, pixel_y))
                        red_values.append(pixel[0])
                        green_values.append(pixel[1])
                        blue_values.append(pixel[2])

            # 计算中值
            median_color = (
                sorted(red_values)[len(red_values) // 2],
                sorted(green_values)[len(green_values) // 2],
                sorted(blue_values)[len(blue_values) // 2]
            )

            # 在新图像中设置中值后的像素
            smoothed_image.putpixel((x, y), median_color)

    it.save_image(smoothed_image, save_path)
    print('中值滤波器成功')
    if show:
        it.compare_image_show(image, smoothed_image)


def sharpen_filter(image, save_path: str, show: bool = False, order: int = 1):
    """
    一阶微分锐化滤波器
    """

    if order not in (1, 2):
        raise ValueError('只能一/二阶')

    width, height = image.size
    # 获取图像像素数据
    pixels = image.load()

    if order == 1:
        sharp_image_1st = Image.new("RGB", (width, height))
        pixels_1st = sharp_image_1st.load()
        # 一阶微分锐化滤波器
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                r = pixels[x + 1, y][0] - pixels[x - 1, y][0]
                g = pixels[x + 1, y][1] - pixels[x - 1, y][1]
                b = pixels[x + 1, y][2] - pixels[x - 1, y][2]
                pixels_1st[x, y] = (r, g, b)
        it.save_image(sharp_image_1st, save_path)
        print('一阶微分锐化成功')
        if show:
            it.compare_image_show(image, sharp_image_1st)

    elif order == 2:
        sharp_image_2nd = Image.new("RGB", (width, height))
        pixels_2nd = sharp_image_2nd.load()
        # 二阶微分锐化滤波器
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                r = 5 * pixels[x, y][0] - pixels[x - 1, y][0] - pixels[x + 1, y][0] - pixels[x, y - 1][0] - \
                    pixels[x, y + 1][0]
                g = 5 * pixels[x, y][1] - pixels[x - 1, y][1] - pixels[x + 1, y][1] - pixels[x, y - 1][1] - \
                    pixels[x, y + 1][1]
                b = 5 * pixels[x, y][2] - pixels[x - 1, y][2] - pixels[x + 1, y][2] - pixels[x, y - 1][2] - \
                    pixels[x, y + 1][2]
                pixels_2nd[x, y] = (r, g, b)
        it.save_image(sharp_image_2nd, save_path)
        print('二阶微分锐化成功')
        if show:
            it.compare_image_show(image, sharp_image_2nd)




if __name__ == "__main__":
    img_path = 'static/image_in/nana.jpg'

    cal1_path = 'static/image_in/eject.png'
    cal2_path = 'static/image_in/no-eject.png'

    save_path = 'static/image_out/test_out3.jpg'
    input_image = Image.open(img_path)

    img1 = Image.open(cal1_path)
    img2 = Image.open(cal2_path)

    # point_op_reverse(input_image, save_path)
    # point_op_log(input_image, save_path=save_path, show=True)
    # point_op_powertrans(input_image, save_path, c=2, gamma=1.8, show=True)
    # point_op_contrast_stretching(input_image, save_path, 50, 200, True)
    # point_op_gray_slice(input_image, save_path, 50, 200, True)
    # point_op_bitplane_slice(input_image, save_path, 7, True)

    # image_cal1(img1, img2, save_path, 'divide', True)
    # image_cal2(input_image, save_path, True)
    # image_cal3(img1, img2, save_path, '^', True)

    # histogram_eq(input_image, save_path, True)

    # linear_smoothing_filter(input_image, 3, save_path, True)
    # middle_smoothing_filter(input_image, 3, save_path, True)
    sharpen_filter(input_image, save_path, True, 1)
