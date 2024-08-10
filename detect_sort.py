import cv2
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
import matplotlib.pyplot as plt
from sort import *

# 初始化 SORT
sort_tracker = Sort()

# 配置 Detectron2
cfg = get_cfg()
config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_name))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
cfg.MODEL.WEIGHTS = (
    "/kaggle/input/detectron2_coincide_separation/pytorch/default/1/model_final.pth"
)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 你的类别数量

predictor = DefaultPredictor(cfg)

# 打开视频文件或相机
cap = cv2.VideoCapture("/kaggle/input/particle-video/output_video.mp4")
# Matplotlib figure for displaying multiple images
fig, ax = plt.subplots(4, 1, figsize=(20, 50))
indices = [ax[0], ax[1], ax[2], ax[3]]
# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定义视频保存的格式和输出路径
out_video = cv2.VideoWriter(
    "output_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (frame_width, frame_height),
)
# 创建保存图像的文件夹
output_image_dir = "result_ini"
os.makedirs(output_image_dir, exist_ok=True)
# 打开 result_ini.txt 文件用于写入检测结果
with open(os.path.join(output_image_dir, "result_ini.txt"), "w") as result_file:
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 Detectron2 进行目标检测
        outputs = predictor(frame)
        # 使用 Visualizer 绘制检测结果
        v = Visualizer(
            frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # 将绘制后的图像转换回 BGR 格式并写入视频
        out_frame = out.get_image()[:, :, ::-1]

        # 在 Matplotlib 中显示图像
        indices[frame_count].imshow(out_frame)
        indices[frame_count].grid(False)
        frame_count += 1

        out_video.write(out_frame)

        # 获取检测框
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        # 获取类别名称
        class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

        # 将框、分数和类别组合成 SORT 所需的输入格式
        detections = np.empty((0, 6))  # 包含类别信息的检测框格式
        for box, score, class_id in zip(boxes, scores, classes):
            detection = np.array([box[0], box[1], box[2], box[3], score, class_id])
            detections = np.vstack((detections, detection))

        # 更新 SORT 跟踪器
        tracks = sort_tracker.update(detections)

        # 绘制检测框和跟踪框
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track[:6]
            class_name = class_names[int(class_id)]  # 获取类别名称
            # 将检测结果写入 result_ini.txt
            result_file.write(
                f"Frame {frame_count}: ID {int(track_id)}, class_id: {int(class_id)}, Class: {class_name}, Box [{x1}, {y1}, {x2}, {y2}]\n"
            )
            print(
                "predict result:",
                f"Frame {frame_count}: ID {int(track_id)}, Class: {class_name}, Box [{x1}, {y1}, {x2}, {y2}]\n",
            )

        # 保存当前帧为图像文件
        frame_filename = os.path.join(output_image_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, out_frame)
        frame_count += 1

        # 使用 Matplotlib 显示图像（可选）
        # frame_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(frame_rgb)
        # plt.axis("off")  # 隐藏坐标轴
        # plt.show()

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
