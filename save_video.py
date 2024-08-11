import cv2, os

# 设置图片文件夹和视频输出路径
image_folder = "result_ini"
video_path = "output_video.avi"

# 获取所有图像文件，假设文件名形式为 "frame_1.jpg", "frame_2.jpg"...
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # 根据数字部分排序

# 读取一张图片以获得尺寸信息
sample_img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = sample_img.shape

# 使用MJPG编解码器
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))  # 帧率设为20

# 逐个添加图像到视频
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video.write(img)

# 释放VideoWriter
video.release()
from IPython.display import FileLink

FileLink(r"output_video.avi")
