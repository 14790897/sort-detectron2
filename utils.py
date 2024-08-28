from pathlib import Path
import cv2, os
import numpy as np
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from save_data import save_data
from detectron2.data import DatasetCatalog, MetadataCatalog


def configure_detectron(data_register_training, data_register_valid):
    # 配置 Detectron2
    cfg = get_cfg()
    config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = (data_register_training,)
    cfg.DATASETS.TEST = (data_register_valid,)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
    cfg.MODEL.WEIGHTS = "./model_final.pth"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        64  # 64 is slower but more accurate (128 faster but less accurate)
    )
    cfg.SOLVER.IMS_PER_BATCH = 2  # (2 is per defaults)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # coincide and separation
    cfg.INPUT.MASK_FORMAT = "bitmask"  # 当指定bitmask的时候,只有rle格式可以使用，指定Polygon的时候只有Polygon格式可以使用
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.SOLVER.BASE_LR = 0.0005  # (quite high base learning rate but should drop)
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.GAMMA = 0.1

    cfg.SOLVER.WARMUP_ITERS = 10  # How many iterations to go from 0 to reach base LR
    cfg.SOLVER.MAX_ITER = 2000  # Maximum of iterations 1
    cfg.SOLVER.STEPS = (500, 1000)  # At which point to change the LR 0.25,0.5
    cfg.TEST.EVAL_PERIOD = 250
    cfg.SOLVER.CHECKPOINT_PERIOD = 250
    cfg.MODEL.DEVICE = "cpu"  # 强制使用 CPU
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def detect_and_track(frame, cfg, sort_tracker, track_history, frame_count, predictor):

    outputs = predictor(frame)
    # 使用 Visualizer 绘制检测结果
    # v = Visualizer(
    #     frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    # )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # # 将绘制后的图像转换回 BGR 格式并写入视频
    # frame = out.get_image()[:, :, ::-1]
    # frame = np.array(frame)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    detections = np.empty((0, 6))
    for box, score, class_id in zip(boxes, scores, classes):
        detection = np.array([box[0], box[1], box[2], box[3], score, class_id])
        detections = np.vstack((detections, detection))

    tracks = sort_tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2 = track[:4]
        class_id = track[4]
        track_id = track[8]
        if class_id >= 0:
            class_name = class_names[int(class_id)][0]
            cv2.putText(
                frame,
                f"{int(track_id)} {class_name}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
            )
            track_id = int(track_id)
            print(
                "predict result:",
                f"Frame {frame_count}: ID {track_id}, Class: {class_name}, Box [{x1}, {y1}, {x2}, {y2}]\n",
            )

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((int(x1), int(y1), int(x2), int(y2)))

            for i in range(1, len(track_history[track_id])):
                pt1 = (
                    track_history[track_id][i - 1][0],
                    track_history[track_id][i - 1][1],
                )
                pt2 = (track_history[track_id][i][0], track_history[track_id][i][1])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            save_data(
                id=track_id,
                class_name=class_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                img=frame,
                frame_count=frame_count,
            )


# 处理视频文件
def process_video(video_path, cfg, sort_tracker, predictor):
    track_history = {}  # 新增：用于存储每个track_id的历史轨迹

    # 打开视频文件或相机
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 定义视频保存的格式和输出路径
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_video = cv2.VideoWriter(
        "output_video.avi",
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    frame_count = 0

    while cap.isOpened():  # and frame_count < 20修改这里，限制帧数为 20:
        ret, frame = cap.read()
        if not ret:
            break
        # 调用检测和跟踪逻辑
        detect_and_track(
            frame, cfg, sort_tracker, track_history, frame_count, predictor
        )

        out_video.write(frame)
        frame_count += 1
    cap.release()
    out_video.release()


# 处理图片集合
def process_images_to_video(images_dir, cfg, sort_tracker, predictor):
    frame_count = 0
    track_history = {}

    # 获取图片路径列表并检查非空
    image_paths = list(sorted(Path(images_dir).glob("*.jpg")))
    if not image_paths:
        print("未找到图片")
        return

    # 读取第一张图片来确定视频尺寸
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        print("无法读取第一张图片")
        return
    frame_height, frame_width = first_frame.shape[:2]

    # 设定输出视频的参数
    fps = 12  # 每秒12帧
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用mp4v编码

    # 创建VideoWriter对象
    out_video = cv2.VideoWriter(
        "output_video_from_img.mp4", fourcc, fps, (frame_width, frame_height)
    )

    # 处理每张图片并写入视频
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"警告: 无法读取 {image_path}，跳过此图片")
            continue
        detect_and_track(
            frame, cfg, sort_tracker, track_history, frame_count, predictor
        )

        out_video.write(frame)  # 写入帧到视频
        frame_count += 1

    # 释放VideoWriter资源
    out_video.release()
    print("视频处理完成，保存至 output_video.mp4")


# 自动检测输入类型并处理
def auto_detect_and_process(input_path, cfg, sort_tracker, predictor):
    if Path(input_path).is_dir():
        print("检测到图片集合")
        process_images_to_video(input_path, cfg, sort_tracker, predictor)
    elif Path(input_path).is_file() and input_path.endswith((".mp4", ".avi")):
        print("检测到视频文件")
        process_video(input_path, cfg, sort_tracker, predictor)
    else:
        print("未知的输入类型，请输入一个视频文件或图片集合的目录")
