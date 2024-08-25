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
    return cfg


# 处理视频文件
def process_video(video_path, cfg, sort_tracker):
    track_history = {}  # 新增：用于存储每个track_id的历史轨迹

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)
    # 打开视频文件或相机
    cap = cv2.VideoCapture("./output_video.mp4")

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
    output_image_dir = "result_ini"
    os.makedirs(output_image_dir, exist_ok=True)
    with open(os.path.join(output_image_dir, "result_ini.txt"), "w") as result_file:
        frame_count = 0

        while cap.isOpened():  # and frame_count < 20修改这里，限制帧数为 20:
            ret, frame = cap.read()
            if not ret:
                break

            # 使用 Detectron2 进行目标检测
            outputs = predictor(frame)
            # 使用 Visualizer 绘制检测结果
            # v = Visualizer(
            #     frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
            # )
            # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # # 将绘制后的图像转换回 BGR 格式并写入视频
            # frame = out.get_image()[:, :, ::-1]
            # frame = np.array(frame)
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
                x1, y1, x2, y2 = track[:4]  # 提取边界框坐标
                class_id = track[4]  # 提取类别 ID
                track_id = track[8]  # 提取跟踪 ID
                if class_id >= 0:
                    class_name = class_names[int(class_id)][0]  # 获取类别名称 首字母
                    print("int(class_id):", int(class_id), "class_name:", class_name)
                    # 将检测结果写入 result_ini.txt ，格式为：track_id class_id x y frame_count
                    result_file.write(
                        "%i %i %i %i %i\n"
                        % (
                            int(track_id),
                            int(class_id),
                            int((x1 + x2) / 2),
                            int((y1 + y2) / 2),
                            frame_count,
                        )
                    )
                    print(
                        "predict result:",
                        f"Frame {frame_count}: ID {int(track_id)}, Class: {class_name}, Box [{x1}, {y1}, {x2}, {y2}]\n",
                    )
                    # 在图像上绘制跟踪ID和类别名称
                    cv2.putText(
                        frame,
                        f" {int(track_id)}  {class_name}",
                        (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        2,
                    )

                    # 保存当前坐标到轨迹历史中
                    track_id = int(track_id)
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((int(x1), int(y1), int(x2), int(y2)))
                    # 绘制轨迹
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
                else:
                    print("class id小于0，class_id", class_id)
            # 在这里添加写入视频文件的代码
            out_video.write(frame)
            # 保存当前帧为图像文件
            # frame_filename = os.path.join(output_image_dir, f"frame_{frame_count:04d}.jpg")
            # cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()
        out_video.release()


# 处理图片集合
def process_images(images_dir, cfg):
    predictor = DefaultPredictor(cfg)
    for image_path in sorted(Path(images_dir).glob("*.jpg")):  # 假设图片扩展名为.jpg
        frame = cv2.imread(str(image_path))
        outputs = predictor(frame)
        # 进行检测和追踪等处理...
        print(f"处理图片：{image_path}")


# 自动检测输入类型并处理
def auto_detect_and_process(input_path, cfg):
    if Path(input_path).is_dir():
        print("检测到图片集合")
        process_images(input_path, cfg)
    elif Path(input_path).is_file() and input_path.endswith((".mp4", ".avi")):
        print("检测到视频文件")
        process_video(input_path, cfg)
    else:
        print("未知的输入类型，请输入一个视频文件或图片集合的目录")
