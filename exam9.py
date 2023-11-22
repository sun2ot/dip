import numpy as np
from scipy.signal import convolve2d
import ImageTools as it
import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

def rgb_image_segmentation1(image, save_path, show: bool=False):
    """
    灰度不连续分割
    """
    # 转换为numpy数组
    img_array = np.array(image)

    # 提取灰度信息
    grayscale_image = np.mean(img_array, axis=-1)

    # 设定阈值，这里可以根据具体情况调整
    threshold = 128

    # 根据阈值进行分割
    segmented_image = np.where(grayscale_image > threshold, 255, 0)

    # 转换为PIL Image对象
    segmented_image = it.fromarray(segmented_image.astype(np.uint8))

    if show:
        it.compare_image_show(image, segmented_image)
    it.save_image(segmented_image, save_path)
    print("灰度不连续分割完成！")


def rgb_image_segmentation2(image, save_path, show: bool=False):
    """
    基于像素阈值的分割
    """
    # 写死阈值
    threshold = (128, 128, 128)
    # 转换为numpy数组
    img_array = np.array(image)

    # 提取R、G、B通道
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]

    # 根据阈值进行分割
    segmented_image = np.where((red_channel > threshold[0]) & (green_channel > threshold[1]) & (blue_channel > threshold[2]), 88, 22)

    # 转换为PIL Image对象
    segmented_image = it.fromarray(segmented_image.astype(np.uint8))

    if show:
        it.compare_image_show(image, segmented_image)
    it.save_image(segmented_image, save_path)
    print("像素阈值分割完成！")


def region_growing(image, save_path, seed=(0,0), show: bool=False):
    """基于区域生长算法的分割"""
    image = image.convert("L")
    img_array = np.array(image)
    height, width = img_array.shape
    visited = np.zeros_like(img_array, dtype=bool)
    out_img = np.zeros_like(img_array)

    # Stack to store the pixel positions.
    stack = []
    stack.append(seed)

    while len(stack) > 0:
        s = stack.pop()
        x, y = s

        if not visited[x, y]:
            visited[x, y] = True
            out_img[x, y] = img_array[x, y]

            # Check all eight neighbours.
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if nx >= 0 and nx < height and ny >= 0 and ny < width:
                        # If the neighbour pixel's intensity is similar to the seed pixel, add it to the stack.
                        if not visited[nx, ny] and abs(int(img_array[nx, ny]) - int(img_array[x, y])) < 20:
                            stack.append((nx, ny))
    result = it.fromarray(out_img)
    if show:
        it.compare_image_show(image, result)
    it.save_image(result, save_path)
    print("区域生长分割完成！")
    return result


def sobel_edge_detection(image, save_path, show: bool=False):
    """sobel算子边缘检测"""
    image = image.convert('L')
    img_array = np.array(image)

    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = np.abs(convolve2d(img_array, sobel_kernel_x, mode='same'))
    gradient_y = np.abs(convolve2d(img_array, sobel_kernel_y, mode='same'))

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    normalized_magnitude = ((gradient_magnitude - gradient_magnitude.min()) /
                            (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)

    edge_image = it.fromarray(normalized_magnitude)

    if show:
        it.compare_image_show(image, edge_image)
    it.save_image(edge_image, save_path)
    print("sobel边缘检测完成！")
    return edge_image


def prewitt_edge_detection(image, save_path, show: bool=False):
    """prewitt算子边缘检测"""
    image = image.convert('L')
    img_array = np.array(image)

    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    gradient_x = np.abs(convolve2d(img_array, prewitt_kernel_x, mode='same'))
    gradient_y = np.abs(convolve2d(img_array, prewitt_kernel_y, mode='same'))

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    normalized_magnitude = ((gradient_magnitude - gradient_magnitude.min()) /
                            (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)

    edge_image = it.fromarray(normalized_magnitude)

    if show:
        it.compare_image_show(image, edge_image)
    it.save_image(edge_image, save_path)
    print("prewitt边缘检测完成！")
    return edge_image


if __name__ == "__main__":

    image = it.read_image('static/image_in/nana.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # rgb_image_segmentation1(image, save_path, True)
    # rgb_image_segmentation2(image, save_path, True)
    # region_growing(image, save_path, show=True)

    # sobel_edge_detection(image,save_path,show=True)
    prewitt_edge_detection(image,save_path,show=True)

