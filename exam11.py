import cv2
import numpy as np
from scipy.spatial import distance
from skimage.feature import local_binary_pattern
import ImageTools as it

# 计算颜色直方图特征
def compute_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 计算LBP特征
def compute_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# 计算欧氏距离
def euclidean_distance(feature1, feature2):
    return distance.euclidean(feature1, feature2)

# 计算余弦相似度
def cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

# 图像检索
def image_retrieval(query_image, database_images):
    query_hist = compute_color_histogram(query_image)
    query_lbp = compute_lbp(query_image)
    results = []

    for image in database_images:
        hist = compute_color_histogram(image)
        lbp = compute_lbp(image)

        # 使用欧氏距离进行特征匹配
        distance_euclidean = euclidean_distance(query_hist, hist)

        # 使用余弦相似度进行特征匹配
        similarity_cosine = cosine_similarity(query_lbp, lbp)

        results.append((image, distance_euclidean, similarity_cosine))

    # 按欧氏距离升序排序
    results.sort(key=lambda x: x[1])

    return results


def main(image, save_path, show: bool=False):
    img_bak = image.copy()
    # 将图像转换为RGB模式
    rgb_image = image.convert('RGB')

    # 将RGB图像转换为NumPy数组
    rgb_array = np.array(rgb_image)

    # 将RGB通道顺序调整为BGR
    bgr_array = rgb_array[:, :, ::-1]

    database_images = [cv2.imread('static/image_in/640white.png'),
                   cv2.imread('static/image_in/grape2.png'), 
                   cv2.imread('static/image_in/640black.png')]

    results = image_retrieval(bgr_array, database_images)    
    image = results[0][0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = it.fromarray(image.astype('uint8'))

    if show:
        it.compare_image_show(img_bak, image)
    it.save_image(image, save_path)
    print('检索成功')


if __name__ == "__main__":
    image = it.read_image('static/image_in/grape.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'
    main(image, save_path, True)