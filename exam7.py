from PIL import Image
import cv2
from io import BytesIO
import numpy as np
from collections import Counter
from heapq import heappush, heappop, heapify
import ImageTools as it
import matplotlib.pyplot as plt


def build_huffman_tree(freq):
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_process(image, save_path, show: bool=False):

    # 确保格式为RGB
    image = image.convert('RGB')

    # 展平为一维数组
    image_array = np.array(image)
    flattened_image = image_array.flatten()
    
    # 计算每个像素值的频率
    pixel_freq = dict(Counter(flattened_image))

    # 构建哈夫曼树
    huffman_tree = build_huffman_tree(pixel_freq)

    # 生成哈夫曼节点
    huffman_codes = {symbol: code for symbol, code in huffman_tree}

    # 利用哈夫曼编码-编码图片
    encoded_image = ''.join(huffman_codes[pixel] for pixel in flattened_image)

    # 解码图片
    decoded_image = []
    code = ''
    reverse_mapping = {code: symbol for symbol, code in huffman_codes.items()}
    for bit in encoded_image:
        code += bit
        if code in reverse_mapping:
            symbol = reverse_mapping[code]
            decoded_image.append(symbol)
            code = ''
    decoded_image = np.array(decoded_image, dtype=np.uint8)
    decoded_image = decoded_image.reshape(image_array.shape)

    # 保存图片
    decoded_image = it.fromarray(decoded_image)
    it.save_image(decoded_image, save_path)
    if show:
        it.compare_image_show(image, decoded_image)
    print('哈夫曼编码压缩完成')

def lzw_compression(image):
    """
    LZW compression for RGB images.
    """
    img = np.array(image)
    flat_img = img.flatten()
    
    # 用256个可能字符初始化字典
    dictionary = {chr(i): i for i in range(256)}
    
    # 初始化变量
    p = chr(flat_img[0])
    result = []
    dict_size = 256
    
    for c in map(chr, flat_img[1:]):
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            result.append(dictionary[p])
            # 将 pc 加入字典
            dictionary[pc] = dict_size
            dict_size += 1
            p = c
    
    if p:
        result.append(dictionary[p])
    
    return result

def lzw_decompression(compressed):
    """
    LZW decompression for RGB images.
    """
    # 用256个可能字符初始化字典
    dictionary = {i: chr(i) for i in range(256)}
    
    # 初始化变量
    w = chr(compressed[0])
    result = [w]
    dict_size = 256
    
    for k in compressed[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        
        result.append(entry)
        
        # 将 w+entry[0] 加入字典
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        
        w = entry
    
    return ''.join(result)

def lzw_process(image, save_path, show: bool=False):

    # 执行 LZW 压缩
    compressed = lzw_compression(image)

    # 执行 LZW 解压缩
    decompressed = lzw_decompression(compressed)

    # 将解压数据还原为图片
    decompressed_img = np.array([ord(c) for c in decompressed], dtype=np.uint8)
    decompressed_img = decompressed_img.reshape(image.size[1], image.size[0], 3)
    decompressed_img = it.fromarray(decompressed_img)
    it.save_image(decompressed_img, save_path)

    if show:
        it.compare_image_show(image, decompressed_img)
    print('LZW 压缩完成')


def dct(img):
    M, N = img.shape
    result = np.zeros_like(img, dtype=float)

    u = np.arange(M).reshape(M, 1)
    v = np.arange(N).reshape(1, N)

    cu = np.sqrt(2) / 2 * np.ones_like(u)
    cu[0] = 1

    cv = np.sqrt(2) / 2 * np.ones_like(v)
    cv[:, 0] = 1

    for i in range(M):
        for j in range(N):
            result[i, j] = np.sum(img * cu * cv * np.cos((2 * i + 1) * u * np.pi / (2 * M)) * np.cos(
                (2 * j + 1) * v * np.pi / (2 * N)
            ))

    return result

def idct(coefficients):
    M, N = coefficients.shape
    result = np.zeros_like(coefficients, dtype=float)

    u = np.arange(M).reshape(M, 1)
    v = np.arange(N).reshape(1, N)

    cu = np.sqrt(2) / 2 * np.ones_like(u)
    cu[0] = 1

    cv = np.sqrt(2) / 2 * np.ones_like(v)
    cv[:, 0] = 1

    for i in range(M):
        for j in range(N):
            result[i, j] = np.sum(cu * cv * coefficients * np.cos((2 * i + 1) * u * np.pi / (2 * M)) * np.cos(
                (2 * j + 1) * v * np.pi / (2 * N)
            ))

    return result


def dct_process(image, save_path, show: bool=False):
    # 确保格式为RGB
    image = image.convert('RGB')
    image_array = np.array(image)
    # 转换为灰度图
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype('float')
    # DCT
    coefficients = dct(img_gray)
    # 进行log处理
    coefficients_log = np.log(abs(coefficients))
    # IDCT
    img_reconstructed = idct(coefficients)
    # it.look_nparray(img_reconstructed)

    result = it.fromarray(it.restore255(img_reconstructed))
    if show:
        it.compare_image_show(image, result)
    it.save_image(result, save_path)
    print('DCT 压缩完成')



if __name__ == "__main__":

    image = it.read_image('static/image_in/grape_small.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # huffman_process(image, save_path, show=True)
    # lzw_process(image, save_path, show=True)

    dct_process(image, save_path, show=True)