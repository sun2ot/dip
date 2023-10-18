from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import ImageTools as it
from exam1 import quantize_image
import exam2 as e2


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
        if request.files['image2']:
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
            c = float(request.form['c'])
            print(f'c: {c}')
            e2.point_op_log(image, save_path, c)
        elif fid == 4:
            # 幂次变换
            c = float(request.form['c'])
            gamma = float(request.form['gamma'])
            print(f'c: {c}, gamma: {gamma}')
            e2.point_op_powertrans(image, save_path, c, gamma)
        elif fid == 5:
            # 对比度拉伸
            min = int(request.form['min'])
            max = int(request.form['max'])
            print(f'min: {min}, max: {max}')
            e2.point_op_contrast_stretching(image, save_path, min, max)
        elif fid == 6:
            # 灰度级切片
            min = int(request.form['min'])
            max = int(request.form['max'])
            print(f'min: {min}, max: {max}')
            e2.point_op_gray_slice(image, save_path, min, max)
        elif fid == 7:
            # 位平面切片
            bit = int(request.form['bit'])
            print(f'bit: {bit}')
            e2.point_op_bitplane_slice(image, save_path, bit)
        elif fid == 8:
            # 代数运算：加减乘除
            op = request.form['op']
            e2.image_cal1(image, image2, save_path, op)
        else:
            raise Exception('no function')
        # flask 的静态资源目录 本地嘛 就这条件 凑合吧
        base_url = 'http://localhost:5000/'
        return jsonify({'success': 'image processed',
                        'save_path': base_url + save_path}), 200

    except Exception as e:
        print('server error:\n', e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)
