from PIL import Image
import os

def read_image(file_path):
    """
    读取图像文件并返回图像对象
    """
    image = Image.open(file_path)
    return image

def save_image(image, file_path):
    """
    保存图像对象到指定文件路径
    """
    image.save(file_path)

def quantize_image(image, levels):
    """
    将图像进行灰度级量化
    image: 输入的图像对象
    levels: 灰度级数量（16、64或128）
    返回量化后的图像对象
    """
    if levels not in [16, 64, 128]:
        raise ValueError("Levels must be 16, 64, or 128.")

    # 将图像转换为灰度图像
    image = image.convert("L")

    # 计算每个灰度级的间隔
    interval = 256 // levels

    # 对每个像素进行量化
    quantized_image = image.point(lambda p: int(p // interval) * interval)
    print('server quantize successfully')

    return quantized_image

if __name__ == "__main__":
    print('main')
    # 示例用法
    # 读取图像
    image = read_image("static/image_in/nana.jpg")

    # 对图像进行灰度级量化（例如，使用64个灰度级）
    quantized_image = quantize_image(image, 64)

    # 保存量化后的图像
    os.makedirs('static/image_out/', exist_ok=True)
    save_image(quantized_image, "static/image_out/quantize_image.jpg")
