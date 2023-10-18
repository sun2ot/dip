from PIL import Image
import numpy as np
# import ImageTools as it
from skimage import io, data, color
import matplotlib.pyplot as plt
import cv2
import imutils
import math
import ImageTools as it


def rgb_cmy(img):
    # rgb转cmy
    r, g, b = cv2.split(img)
    r = r/255.0
    g = g/255.0
    b = b/255.0
    c = 1-r
    m = 1-g
    y = 1-b
    result = cv2.merge((c, m, y))
    return result


def cmy_rgb(img):
    c, m, y = cv2.split(img)
    c = c/255.0
    m = m/255.0
    y = y/255.0
    r = 1-c
    g = 1-m
    b = 1-y
    result = cv2.merge((r, g, b))
    return result


def rgb_hsi(rgb_Img):
    img_rows = int(rgb_Img.shape[0])
    img_cols = int(rgb_Img.shape[1])
    b, g, r = cv2.split(rgb_Img)
    r = r/255
    g = g/255
    b = b/255
    hsi_Img = rgb_Img.copy()
    H, S, I = cv2.split(hsi_Img)
    for i in range(img_rows):
        for j in range(img_cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2 +
                          (r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            theta = float(np.arccos(num/den))
            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = math.pi - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1-3*min_RGB/sum
            H = H/(math.pi)
            I = sum/3.0
            # 输出HSI图像，扩展到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_Img[i, j, 0] = H*255
            hsi_Img[i, j, 1] = S*255
            hsi_Img[i, j, 2] = I * 255
    return hsi_Img


def hsi_rgb(hsi_img):
    img_rows = int(hsi_img.shape[0])
    img_cols = int(hsi_img.shape[1])
    H, S, I = cv2.split(hsi_img)
    # normalization[0,1]
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    B, G, R = cv2.split(bgr_img)
    for i in range(img_rows):
        for j in range(img_cols):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) / math.cos(
                        (60 - H[i, j]) * math.pi / 180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) / math.cos(
                        (60 - H[i, j]) * math.pi / 180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) / math.cos(
                        (60 - H[i, j]) * math.pi / 180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = B * 255
            bgr_img[i, j, 1] = G * 255
            bgr_img[i, j, 2] = R * 255
    return bgr_img

# 直方图均衡化


def histogram_eq(input_image, save_path: str, show: bool = True):

    input_image = input_image.convert('L')

    image = np.array(input_image)  # 这里假设你有一张灰度图像

    # 计算直方图
    histogram, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累积分布函数
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()

    # 使用CDF重新映射像素值
    equalized_image = np.interp(
        image, bins[:-1], cdf_normalized).astype(np.uint8)
    equalized_image = Image.fromarray(equalized_image)

    it.save_image(equalized_image, save_path)
    print('直方图均衡化成功')
    if show:
        it.compare_image_show(image, equalized_image)


if __name__ == "__main__":

    img = cv2.imread(("static/image_in/cat.jpg"))
    # RGB-CYM
    img_CMY = rgb_cmy(img)
    img_NER = img_CMY*255
    cv2.imwrite('static/image_out/img_CMY.PNG', img_NER)
    cv2.imshow("CMY image", imutils.resize(img_CMY, 666))
    cv2.imshow("original image", imutils.resize(img, 666))

    # CYM-RGB
    img_cmy = cmy_rgb(img)
    cv2.imshow("RGB image", imutils.resize(img_cmy, 666))
    # cv2.imshow("original image", imutils.resize(img, 666))

    # RGB-HSI
    rgb_Img = cv2.imread("static/image_in/singer.jpg")
    hsi_Img = rgb_hsi(rgb_Img)
    cv2.imwrite('static/image_out/img_HSI.PNG', hsi_Img)

    cv2.imshow("original_1 image", imutils.resize(rgb_Img, 600))
    cv2.imshow("HSI image", imutils.resize(hsi_Img, 600))

    # HSI-RGB
    hsi_Img = cv2.imread("static/image_out/img_HSI.PNG")
    rgb_Img = hsi_rgb(hsi_Img)
    cv2.imshow('RGB_1 image', imutils.resize(hsi_Img, 600))
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 直方图均衡化
    img_path = 'static/image_in/yellwo.jpg'
    input_image = Image.open(img_path)
    save_path = 'static/image_out/eq.jpg'

    histogram_eq(input_image, save_path, True)
