from PIL import Image, ImageDraw
import numpy as np
import ImageTools as it

def get_chain_code(img):
    """链码表示"""
    # convert it to grayscale
    img = img.convert("L")
    img_array = np.array(img)

    # Find the starting point of the contour (assuming it's the top-left corner)
    start_point = None
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if img_array[i, j] > 0:
                start_point = (i, j)
                break
        if start_point:
            break

    if not start_point:
        print("No contour found in the image.")
        return

    # Define the 8-connectivity neighbors
    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Define the chain code
    chain_code = []

    # Get the initial position
    current_position = start_point

    # Mark the visited pixels
    visited = np.zeros_like(img_array, dtype=bool)

    # Start the chain code
    while True:
        for i, neighbor in enumerate(neighbors):
            neighbor_position = (current_position[0] + neighbor[0], current_position[1] + neighbor[1])

            # Check if the neighbor position is within the image boundaries and not visited
            if (
                0 <= neighbor_position[0] < img_array.shape[0] and
                0 <= neighbor_position[1] < img_array.shape[1] and
                img_array[neighbor_position] > 0 and
                not visited[neighbor_position]
            ):
                chain_code.append(i)
                current_position = neighbor_position
                visited[current_position] = True
                break

        # Check if we have returned to the starting point
        if current_position == start_point:
            break
    print('链码计算完成')
    return chain_code


# -----------------------------------------------

def edge_detection(image, threshold=10):
    image_array = np.array(image)
    edges = np.zeros_like(image_array)
    edges[np.where(image_array > threshold)] = 255
    return Image.fromarray(edges.astype(np.uint8))


def draw_polygon(image, vertices):
    draw = ImageDraw.Draw(image)
    draw.polygon(vertices.flatten().tolist(), outline=255, fill=0)
    del draw


def polygon_approximation(image, num_vertices, save_path, show: bool=False):
    gray_image = image.convert('L')
    edge_image = edge_detection(gray_image)
    # 获取边缘点
    edge_points = np.column_stack(np.where(np.array(edge_image) > 0))

    # 随机选择初始顶点
    vertices = edge_points[np.random.choice(len(edge_points), num_vertices, replace=False)]

    for _ in range(10): 
        # 将每个边点指定到最近的顶点
        distances = np.linalg.norm(edge_points[:, None, :] - vertices, axis=2)
        closest_vertices = np.argmin(distances, axis=1)

        # 将顶点更新为指定点的平均值
        for i in range(num_vertices):
            if len(edge_points[closest_vertices == i]) > 0:
                vertices[i] = np.mean(edge_points[closest_vertices == i], axis=0)
    
    # 绘制多边形并保存结果图像
    result_image = Image.new("L", gray_image.size, 0)
    draw_polygon(result_image, vertices)
    if show:
        it.compare_image_show(gray_image, result_image)
    it.save_image(result_image, save_path)
    print('多边形近似完成')


# ------------------------------------------------

def binarize_image(img_array, threshold=128):
    binary_img = img_array.copy()
    binary_img[binary_img <= threshold] = 0
    binary_img[binary_img > threshold] = 255
    return binary_img


def region_skeleton(image, save_path, show: bool=False):
    image = image.convert('L')
    image_array = np.array(image)
    binary_img = binarize_image(image_array)
    skeleton = binary_img.copy()

    while True:
        temp_skeleton = skeleton.copy()

        # 对每个非零像素进行骨架化处理
        for i in range(1, binary_img.shape[0] - 1):
            for j in range(1, binary_img.shape[1] - 1):
                if skeleton[i, j] == 255:
                    neighbors = binary_img[i-1:i+2, j-1:j+2].flatten()
                    neighbors_count = np.sum(neighbors) // 255

                    # 判断是否为端点
                    if neighbors_count == 2:
                        if neighbors[1] == 0 or neighbors[3] == 0 or neighbors[5] == 0 or neighbors[7] == 0:
                            temp_skeleton[i, j] = 0

        if np.array_equal(temp_skeleton, skeleton):
            break
        else:
            skeleton = temp_skeleton
    skeleton_img  = it.fromarray(skeleton)
    if show:
        it.compare_image_show(image, skeleton_img)
    it.save_image(skeleton_img, save_path)
    print('骨架化完成！')


# --------------------------------------------------


def calculate_boundary(image):
    # 打开图像并转换为灰度图像
    image = image.convert('L')
    
    # 将图像转换为numpy数组
    image_array = np.array(image)
    
    # 将像素值大于128的设为1，小于等于128的设为0，得到二值图像
    binary_image = np.where(image_array > 128, 1, 0)
    
    # 计算图像边界的周长
    boundary_perimeter = calculate_perimeter(binary_image)
    
    # 计算图像边界的直径
    boundary_diameter = calculate_diameter(binary_image)
    
    return boundary_perimeter, boundary_diameter

def calculate_perimeter(binary_image):
    # 使用8邻域算法计算边界周长
    height, width = binary_image.shape
    perimeter = 0
    
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1:
                # 判断当前像素的8邻域是否有0，如果有则边界周长加1
                if (i == 0 or binary_image[i-1, j] == 0) or \
                   (i == height-1 or binary_image[i+1, j] == 0) or \
                   (j == 0 or binary_image[i, j-1] == 0) or \
                   (j == width-1 or binary_image[i, j+1] == 0) or \
                   (i == 0 or j == 0 or binary_image[i-1, j-1] == 0) or \
                   (i == 0 or j == width-1 or binary_image[i-1, j+1] == 0) or \
                   (i == height-1 or j == 0 or binary_image[i+1, j-1] == 0) or \
                   (i == height-1 or j == width-1 or binary_image[i+1, j+1] == 0):
                    perimeter += 1
    
    return perimeter

def calculate_diameter(binary_image):
    # 使用最大内接圆直径计算边界直径
    # 使用膨胀操作将边界膨胀成一个实心圆，直径即为膨胀操作的半径乘以2
    from scipy.ndimage import binary_dilation
    
    dilated_image = binary_dilation(binary_image)
    diameter = np.sum(dilated_image) * 2
    
    return diameter


if __name__ == "__main__":

    image = it.read_image('static/image_in/nana.jpg')
    save_path = 'static/image_out/' + it.gen_timestamp_name() + '.jpg'

    # code = get_chain_code(image)
    # print("Chain Code:", code)

    # 多边形近似
    num_vertices = 3  # 可根据需要调整多边形的顶点数
    # vertices = polygon_approximation(image, num_vertices, save_path, True)

    # skeleton = region_skeleton(image, save_path, True)

    # 计算边界描述子
    perimeter, diameter = calculate_boundary(image)
    print("边界周长:", perimeter)
    print("边界直径:", diameter)

