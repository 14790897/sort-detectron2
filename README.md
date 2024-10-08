[English](./README_en.md)

# Detectron2 与 SORT 跟踪算法的实现

本仓库包含了一个 Python 脚本，集成了 Detectron2 用于目标检测，并结合 SORT 算法进行目标跟踪。脚本读取视频文件，处理每一帧以检测目标，并跟踪这些目标在帧之间的运动，最后将结果保存为一个带有注释的跟踪视频。

## Kaggle Notebook

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-blue?style=flat&logo=kaggle)](https://www.kaggle.com/code/liuweiq/sort-detectron2)

## 环境依赖

在运行脚本之前，请确保已安装以下依赖：

- Python 3.6 及以上版本
- OpenCV
- PyTorch
- Detectron2
- SORT (Simple Online and Realtime Tracking) 库

### 安装步骤

1. **克隆仓库：**

   ```bash
   git clone https://github.com/yourusername/sort-detectron2.git
   cd sort-detectron2
   ```

2. **安装依赖：**

   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio
   pip install opencv-python
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.9/index.html
   pip install numpy matplotlib
   ```

3. **下载 Detectron2 预训练权重：**

   确保 `model_final.pth` 文件位于工作目录中，或者修改脚本中的路径以指向你的预训练模型。
   我的权重文件地址：[kaggle download](https://storage.googleapis.com/kagglesdsdata/models/77237/92130/model_final.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240819%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240819T082352Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3ad5df116a3c4ad30f12b34562b598b0f723edbf4fc3df2c96806e63ed99944e676501f9f155ea4dbe972490dca9067e934dd5bc2c691f86dc3f5f31ef2059b1c88fa0f9c4dc55b03893f8a6c93a94f4a0379a4356deae08107ec1f6f8a1946bdd6958dcb3297fa10a9aa5d5be030cc4aca2b4c5237ea56bf2242a0893df5b0f24089352e0bd96def2d7be73d3cf47a572581b70fb2c48c01defd9ad5b83f4ca725af701183f4dd09549b52f8475b7538640bdf9cffb44dfd61691cc64d7261e71e6fd907955f99a8e43a1f739418f87034bfd8a6c7ba9e3224c40ef864f51c63bba8adf2aeb85f48ebb8f2d4d9329bdffdf6edd942e21d2f39fadb5d82ba9f7)
   [kaggle model page](https://www.kaggle.com/models/liuweiq/detectron2_coincide_separation)

## 使用方法

### 运行脚本

要处理视频并进行目标跟踪，请运行：

```bash
python main.py
```

### 配置

- **输入视频**：默认情况下，脚本从 `./output_video.mp4` 读取视频。可以修改路径以处理不同的视频文件。
- **输出**：脚本输出带有注释的 `output_video.avi` 视频，并将结果保存在 `result_ini` 目录下的 `result_ini.txt` 文件中。
- **帧数限制**：脚本默认处理整个视频。如果只处理前 20 帧，可以取消注释并修改 `while` 循环条件。

### 输出文件

- **注释视频**：脚本将生成一个带有检测目标和跟踪 ID 的输出视频文件 `output_video.avi`。
- **跟踪数据**：跟踪数据保存在 `result_ini.txt` 文件中，格式为：
  ```
  track_id class_id x y frame_count
  ```
  其中 `x` 和 `y` 为跟踪目标中心的坐标。

## 关键组件

### SORT 跟踪器

使用 SORT 算法来对检测到的目标进行逐帧跟踪。通过卡尔曼滤波来预测目标的未来位置，并更新轨迹。

### Detectron2

使用 Detectron2 进行目标检测。此实现使用了带有 ResNet-50 骨干网络的 Mask R-CNN 模型。可以根据需要修改配置，使用不同的模型或设置。

### 可视化工具

可视化工具负责在每一帧视频上绘制检测和跟踪信息。在保存之前，图像上会标注出边界框和跟踪 ID。

## 注意事项

- **CPU 使用**：脚本默认在 CPU 上运行。若有 GPU 并希望使用，请将 `cfg.MODEL.DEVICE` 修改为 `"cuda"`。
- **自定义**：您可以在脚本配置部分轻松自定义模型、数据集和参数，以适应不同的场景或数据集。

## 许可证

本项目采用 MIT 许可证。详细信息请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- 感谢 Facebook AI Research (FAIR) 提供的 **Detectron2** 框架。
- 感谢 Alex Bewley 提供的 **SORT** 目标跟踪算法。

欢迎对本项目进行修改和适应，亦欢迎任何贡献！


### 以后可以参考
https://github.com/sayef/detectron2-deepsort-pytorch/blob/master/demo_detectron2_deepsort.py