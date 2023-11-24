from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import ImageTools as it
from exam1 import quantize_image
import exam2 as e2
import exam3 as e3
import exam4 as e4
import exam5 as e5
import exam6 as e6
import exam7 as e7
import exam8 as e8
import exam9 as e9
import exam10 as e10
import exam11 as e11


app = Flask(__name__)
cors = CORS(app, resources={
            r"/process_image": {"origins": "http://localhost:*"}})


"""
1. 只是一个课程作业而已，没必要加多个路由模块了，直接前端传参区分得了......
2. 参数有效性验证能省则省吧，反正前端有alert，也看不出来啥
"""


@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # 从请求中获取上传的图像文件
        # 将 FileStorage 对象的内容读取为字节流
        uploaded_image = request.files['image']
        image_bytes = BytesIO(uploaded_image.read())
        image = Image.open(image_bytes)
        uploaded_image2 = None
        image2 = None
        if len(request.files) == 2:
            print('has 2 images')
            uploaded_image2 = request.files['image2']
            image_bytes2 = BytesIO(uploaded_image2.read())
            image2 = Image.open(image_bytes2)

        # 功能序号
        fid = int(request.form['fid'])
        print(f'fid is: {fid}')

        # 最终文件路径 + 文件名
        os.makedirs('static/image_out', exist_ok=True)
        # save_path = os.path.join('static/image_out', it.gen_timestamp_name() + '.jpg')
        url_parts = ['static/image_out', it.gen_timestamp_name() + '.jpg']
        save_path = '/'.join(url_parts)
        print(f'result is: {save_path}')

        # 依据前端传入的fid=n参数，判断调用exam-n中的函数
        if fid == 1:
            # 灰度级量化
            levels = int(request.form['levels'])
            print(f'target level: {levels}')
            # 对图像进行灰度级量化
            quantized_image = quantize_image(image, levels)
            it.save_image(quantized_image, save_path)
        elif fid == 2:
            # 反转变换
            e2.point_op_reverse(image, save_path)
        elif fid == 3:
            # 对数变换
            c = float(request.form['input1'])
            print(f'c: {c}')
            e2.point_op_log(image, save_path, c)
        elif fid == 4:
            # 幂次变换
            c = float(request.form['input1'])
            gamma = float(request.form['input2'])
            print(f'c: {c}, gamma: {gamma}')
            e2.point_op_powertrans(image, save_path, c, gamma)
        elif fid == 5:
            # 对比度拉伸
            min = int(request.form['input1'])
            max = int(request.form['input2'])
            print(f'min: {min}, max: {max}')
            e2.point_op_contrast_stretching(image, save_path, min, max)
        elif fid == 6:
            # 灰度级切片
            min = int(request.form['input1'])
            max = int(request.form['input2'])
            print(f'min: {min}, max: {max}')
            e2.point_op_gray_slice(image, save_path, min, max)
        elif fid == 7:
            # 位平面切片
            bit = int(request.form['input1'])
            print(f'bit: {bit}')
            e2.point_op_bitplane_slice(image, save_path, bit)
        elif fid == 8:
            # 代数运算：加减乘除
            op = request.form['input1']
            print(f'op: {op}')
            e2.image_cal1(image, image2, save_path, op)
        elif fid == 9:
            # 代数运算：非运算
            e2.image_cal2(image, save_path)
        elif fid == 10:
            # 代数运算：与、或、异或运算
            op = request.form['input1']
            print(f'op: {op}')
            e2.image_cal3(image, image2, save_path, op)
        elif fid == 11:
            # 直方图均衡化
            e2.histogram_eq(image, save_path)
        elif fid == 12:
            # 线性平滑滤波器
            kernel_size = int(request.form['input1'])
            print(f'kernel_size: {kernel_size}')
            e2.linear_smoothing_filter(image, kernel_size, save_path)
        elif fid == 13:
            # 中值滤波器
            kernel_size = int(request.form['input1'])
            print(f'kernel_size: {kernel_size}')
            e2.middle_smoothing_filter(image, kernel_size, save_path)
        elif fid == 14:
            # 锐化滤波器
            order = int(request.form['input1'])
            print(f'order: {order}')
            e2.sharpen_filter(image, save_path, order)
        elif fid == 15:
            # 以下四个为彩色空间变换
            e3.rgb2cmy(image, save_path)
        elif fid == 16:
            e3.rgb2hsi(image, save_path)
        elif fid == 17:
            e3.rgb2yuv(image, save_path)
        elif fid == 18:
            e3.rgb2ycbcr(image, save_path)
        elif fid == 19:
            # 补色
            e3.com_color(image, save_path)
        elif fid == 20:
            # 灰度变换
            alpha = int(request.form['input1'])
            beta = int(request.form['input2'])
            e3.grayscale_transform(image, alpha, beta, save_path)
        elif fid == 21:
            # 傅里叶变换
            e4.fourier_transform(image, save_path)
        elif fid == 22:
            # 快速傅里叶变换
            e4.fast_fourier_transform(image, save_path)
        elif fid == 23:
            # 理想低通滤波器
            cf = int(request.form['input1'])
            e5.ideal_low_filter(image, cf, save_path)
        elif fid == 24:
            # 巴特沃斯低通滤波器
            cf = int(request.form['input1'])
            order = int(request.form['input2'])
            e5.butterworth_low_filter(image, cf, order, save_path)
        elif fid == 25:
            # 理想高通滤波器
            cf = int(request.form['input1'])
            e5.ideal_high_filter(image, cf, save_path)
        elif fid == 26:
            # Butterworth高通滤波器
            cf = int(request.form['input1'])
            order = int(request.form['input2'])
            e5.butterworth_high_filter(image, cf, order, save_path)
        elif fid == 27:
            # 添加高斯噪声
            mean = int(request.form['input1'])
            std = int(request.form['input2'])
            e6.add_gaussian_noise(image, save_path, mean, std)
        elif fid == 28:
            # 添加椒盐噪声
            salt_prob = float(request.form['input1'])
            pepper_prob = float(request.form['input2'])
            e6.add_salt_and_pepper_noise(
                image, save_path, salt_prob, pepper_prob)
        elif fid == 29:
            # 添加波动噪声
            intensity = int(request.form['input1'])
            e6.add_uniform_noise(image, save_path, intensity)
        elif fid == 30:
            # 算术均值滤波器
            kernel_size = int(request.form['input1'])
            e6.ArithmeticMean_filter(image, kernel_size, save_path)
        elif fid == 31:
            # 中值滤波器
            kernel_size = int(request.form['input1'])
            e6.Median_filter(image, kernel_size, save_path)
        elif fid == 32:
            # 高斯滤波器
            kernel_size = int(request.form['input1'])
            sigma = int(request.form['input2'])
            e6.Gaussian_filter(image, kernel_size, sigma, save_path)
        elif fid == 33:
            # 最大值滤波器
            kernel_size = int(request.form['input1'])
            e6.Max_filter(image, kernel_size, save_path)
        elif fid == 34:
            # 最小值滤波器
            kernel_size = int(request.form['input1'])
            e6.Min_filter(image, kernel_size, save_path)
        elif fid == 35:
            # 带阻滤波器
            center_frequency = int(request.form['input1'])
            bandwidth = int(request.form['input2'])
            e6.bandstop_filter(image, center_frequency, bandwidth, save_path)
        elif fid == 36:
            # 带通滤波器
            low_frequency = int(request.form['input1'])
            high_frequency = int(request.form['input2'])
            e6.bandpass_filter(image, low_frequency, high_frequency, save_path)
        elif fid == 37:
            # 陷波滤波器
            center_frequency = int(request.form['input1'])
            bandwidth = int(request.form['input2'])
            e6.notch_filter(image, center_frequency, bandwidth, save_path)
        elif fid == 38:
            # 哈夫曼编码
            e7.huffman_process(image, save_path)
        elif fid == 39:
            # LZW编码
            e7.lzw_process(image, save_path)
        elif fid == 40:
            # DCT有损压缩
            e7.dct_process(image, save_path)
        elif fid == 41:
            # DFT有损压缩
            e7.dft_process(image, save_path)
        elif fid == 42:
            # 膨胀操作
            e8.dilate(image, save_path)
        elif fid == 43:
            # 腐蚀操作
            e8.erode(image, save_path)
        elif fid == 44:
            # 开操作
            e8.open(image, save_path)
        elif fid == 45:
            # 闭操作
            e8.close(image, save_path)
        elif fid == 46:
            # 边界提取
            e8.boundary_extract(image, save_path)
        elif fid == 47:
            # 区域填充
            e8.region_fill(image, save_path)
        elif fid == 48:
            # 灰度不连续分割
            e9.rgb_image_segmentation1(image, save_path)
        elif fid == 49:
            # 基于像素阈值的分割
            e9.rgb_image_segmentation2(image, save_path)
        elif fid == 50:
            # 区域生长
            e9.region_growing(image, save_path)
        elif fid == 51:
            # sobel算子边缘检测
            e9.sobel_edge_detection(image, save_path)
        elif fid == 52:
            # prewitt算子边缘检测
            e9.prewitt_edge_detection(image, save_path)
        elif fid == 53:
            # 链码表示
            chain_code = e10.get_chain_code(image)
            return jsonify({'success': 'image processed',
                'result': chain_code}), 200
        elif fid == 54:
            # 多边形近似
            num_vertices = int(request.form['input1'])
            e10.polygon_approximation(image, num_vertices, save_path)
        elif fid == 55:
            # 区域骨架
            e10.region_skeleton(image, save_path)
        elif fid == 56:
            # 边界描述
            perimeter, diameter = e10.calculate_boundary(image)
            result = f'边界周长: {perimeter} 边界直径: {diameter}'
            return jsonify({'success': 'image processed',
                'result': result}), 200
        elif fid == 57:
            # 图像检索
            e11.main(image, save_path)
        else:
            raise Exception('no function mapped')
        # flask 的静态资源目录 本地嘛 就这条件 凑合吧
        base_url = 'http://localhost:5000/'
        return jsonify({'success': 'image processed',
                        'save_path': base_url + save_path}), 200

    except Exception as e:
        print('-----server error-----\n', e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)
