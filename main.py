# 这里最好有那个detectron2的专属显示器
from detectron2.data import DatasetCatalog, MetadataCatalog
from pathlib import Path
from utils import configure_detectron, auto_detect_and_process
from detectron2.data.datasets import register_coco_instances
from sort import *

Data_Resister_training = "coincide_separation_train"
Data_Resister_valid = "coincide_separation_valid"

# 初始化 SORT
sort_tracker = Sort()
dataDir_train = Path("./coco-data/images")

register_coco_instances(
    Data_Resister_training, {}, "./coco-data/train.json", dataDir_train
)
register_coco_instances(Data_Resister_valid, {}, "./coco-data/val.json", dataDir_train)

metadata = MetadataCatalog.get(Data_Resister_training)
dataset_valid = DatasetCatalog.get(Data_Resister_valid)
dataset_train = DatasetCatalog.get(Data_Resister_training)

print("class name in detectron2: ", metadata.thing_classes)

if __name__ == "__main__":
    data_resister_training = "coincide_separation_train"
    data_resister_valid = "coincide_separation_valid"
    predictor, cfg = configure_detectron(data_resister_training, data_resister_valid)
    input_path = (
        "input_video.mp4"  # 可以是视频也可以是文件夹 images_x ./input_video.mp4
    )
    auto_detect_and_process(input_path, cfg, sort_tracker, predictor)
