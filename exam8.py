import numpy as np
import ImageTools as it
from PIL import Image
import matplotlib.pyplot as plt



def dilate(img, save_path: str='', show: bool=False):
    """膨胀操作"""
    pixels = img.load()
    width, height = img.size
    result = Image.new('RGB', (width, height))
    result_pixels = result.load()
    # 定义膨胀腐蚀操作的核
    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]

    for x in range(width):
        for y in range(height):
            max_r, max_g, max_b = 0, 0, 0
            for i in range(len(kernel)):
                for j in range(len(kernel[0])):
                    new_x = x + i - 1
                    new_y = y + j - 1
                    if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
                        r, g, b = pixels[new_x, new_y]
                        max_r = max(max_r, r)
                        max_g = max(max_g, g)
                        max_b = max(max_b, b)
            result_pixels[x, y] = (max_r, max_g, max_b)
    
    if save_path != '':
        it.save_image(result, save_path)
    if show:
        it.compare_image_show(img, result)
    print("膨胀操作完成！")
    return result


def erode(img, save_path: str='', show: bool=False):
    """腐蚀操作"""
    pixels = img.load()
    width, height = img.size
    result = Image.new('RGB', (width, height))
    result_pixels = result.load()
    # 定义膨胀腐蚀操作的核
    kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    
    for x in range(width):
        for y in range(height):
            min_r, min_g, min_b = 255, 255, 255
            for i in range(len(kernel)):
                for j in range(len(kernel[0])):
                    new_x = x + i - 1
                    new_y = y + j - 1
                    if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
                        r, g, b = pixels[new_x, new_y]
                        min_r = min(min_r, r)
                        min_g = min(min_g, g)
                        min_b = min(min_b, b)
            result_pixels[x, y] = (min_r, min_g, min_b)

    if save_path != '':
        it.save_image(result, save_path)
    if show:
        it.compare_image_show(img, result)
    print("腐蚀操作完成！")
    return result


def opening(img, save_path, show: bool=False):
    """开操作"""
    erosion_img = erode(img)
    opening_img = dilate(erosion_img)
    it.save_image(opening_img, save_path)
    if show:
        it.compare_image_show(img, opening_img)
    print("开操作完成！")
    return opening_img


def closing(img, save_path, show: bool=False):
    """闭操作"""
    dilation_img = dilate(img)
    closing_img = erode(dilation_img)
    it.save_image(closing_img, save_path)
    if show:
        it.compare_image_show(img, closing_img)
    print("闭操作完成！")
    return closing_img


def boundary_extraction(img, save_path: str='', show: bool=False):
    """边界提取"""
    # 进行腐蚀和膨胀操作
    eroded = erode(img)
    dilated = dilate(img)

    # 计算边界
    pixels_eroded = eroded.load()
    pixels_dilated = dilated.load()
    width, height = img.size
    result = Image.new('RGB', (width, height))
    result_pixels = result.load()

    for x in range(width):
        for y in range(height):
            if pixels_eroded[x, y] != pixels_dilated[x, y]:
                result_pixels[x, y] = img.getpixel((x, y))
    
    if save_path != '':
        it.save_image(result, save_path)
    if show:
        it.compare_image_show(img, result)
    print("边界提取完成！")
    return result


def region_filling(image, save_path, seed=(50, 50), new_color=(46, 232, 0), show: bool=False):
    """区域填充"""
    width, height = image.size
    image_np = np.array(image)
    stack = [seed]  # 使用一个栈来保存需要处理的像素位置

    old_color = image_np[seed[1], seed[0]].tolist()  # 获取种子点的颜色

    # 如果种子点的颜色已经是目标颜色，那么没有必要进行填充
    if old_color == new_color:
        return

    while stack:
        x, y = stack.pop()
        if (x < 0) or (x >= width) or (y < 0) or (y >= height):
            continue  # 如果点不在图像范围内，那么跳过

        current_color = image_np[y, x].tolist()
        if current_color == old_color:
            image_np[y, x] = new_color
            stack.append((x, y-1))  # 上
            stack.append((x+1, y))  # 右
            stack.append((x, y+1))  # 下
            stack.append((x-1, y))  # 左

    filled_image = Image.fromarray(np.uint8(image_np))
    it.save_image(filled_image, save_path)
    if show:
        it.compare_image_show(image, filled_image)
    print("区域填充完成！")
    return filled_image



if __name__ == '__main__':

    image = it.read_image('static/image_in/nana.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # dilate(image, save_path, True)
    # erode(image, save_path, True)

    # opening(image, save_path, True)
    # closing(image, save_path, True)

    # boundary_extraction(image, save_path, True)
    region_filling(image, (50, 50), (46, 232, 0), save_path, True)