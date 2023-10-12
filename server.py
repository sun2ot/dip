from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import time
import os
import ImageTools as it
from exam1 import quantize_image
import exam2 as e2


app = Flask(__name__)
cors = CORS(app, resources={r"/process_image": {"origins": "http://localhost:5175"}})

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # 从请求中获取上传的图像文件和灰度级数量
        uploaded_image = request.files['image']
        print(f'upload image: {uploaded_image}')
        levels = int(request.args.get('levels'))
        print(f'target level: {levels}')

        # 确保上传了图像文件
        if not uploaded_image:
            return jsonify({'error': 'No image uploaded'}), 400

        # 读取上传的图像文件
        # 将 FileStorage 对象的内容读取为字节流
        image_bytes = BytesIO(uploaded_image.read())
        image = Image.open(image_bytes)

        # 对图像进行灰度级量化
        quantized_image = quantize_image(image, levels)
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        os.makedirs('static/image_out', exist_ok=True)
        file_path = os.path.join('static/image_out', time.strftime('%H-%M-%S') + '.jpg')
        print(f'save path is: {file_path}')
        it.save_image(quantized_image, file_path)

        print('server process successfully!')
        # 正常来说这里的filepath肯定是url，但本地嘛，没这个条件
        baseURL = 'http://localhost:5000/'
        return jsonify({'success': 'image processed',
                        'file_path': baseURL + file_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
