# Introduction

NMU 数字图像处理作业

# 后端适配要求

1. 函数需要传参 `save_path` 
2. 确保最后生成的图像，通过 `ImageTools.save_image(image, save_path)` 保存

确保以上两点，即可利用[dip-react-ts](https://github.com/sun2ot/dip-react-ts)作为前端

# 环境部署

```
cd dip
pip install -r requirements.txt 或者 conda install --file requirements.txt
python server.py
```