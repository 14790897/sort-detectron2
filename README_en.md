# Detectron2 with SORT Tracker Implementation

This repository contains a Python script that integrates Detectron2 for object detection with the SORT algorithm for object tracking. The code reads a video file, processes each frame to detect objects, and tracks the movement of these objects across frames, saving the results into a video file with annotated object tracks.

## Kaggle Notebook

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-blue?style=flat&logo=kaggle)](https://www.kaggle.com/your-username/your-notebook)

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.6 or above
- OpenCV
- PyTorch
- Detectron2
- SORT (Simple Online and Realtime Tracking) library

### Installation Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/sort-detectron2.git
   cd sort-detectron2
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio
   pip install opencv-python
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.9/index.html
   pip install numpy matplotlib
   ```

3. **Download pre-trained weights for Detectron2:**

   Ensure you have the `model_final.pth` file in the working directory or modify the path in the script to point to your pre-trained model.
   my pth address:[kaggle download](https://storage.googleapis.com/kagglesdsdata/models/77237/92130/model_final.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240819%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240819T082352Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3ad5df116a3c4ad30f12b34562b598b0f723edbf4fc3df2c96806e63ed99944e676501f9f155ea4dbe972490dca9067e934dd5bc2c691f86dc3f5f31ef2059b1c88fa0f9c4dc55b03893f8a6c93a94f4a0379a4356deae08107ec1f6f8a1946bdd6958dcb3297fa10a9aa5d5be030cc4aca2b4c5237ea56bf2242a0893df5b0f24089352e0bd96def2d7be73d3cf47a572581b70fb2c48c01defd9ad5b83f4ca725af701183f4dd09549b52f8475b7538640bdf9cffb44dfd61691cc64d7261e71e6fd907955f99a8e43a1f739418f87034bfd8a6c7ba9e3224c40ef864f51c63bba8adf2aeb85f48ebb8f2d4d9329bdffdf6edd942e21d2f39fadb5d82ba9f7)
   [kaggle model page](https://www.kaggle.com/models/liuweiq/detectron2_coincide_separation)

## Usage

### Running the Script

To process a video and track objects:

```bash
python main.py
```

### Configuration

- **Input Video:** The script reads from `./output_video.mp4` by default. Modify the path to process a different video file.
- **Output:** The script outputs an annotated video `output_video.avi` and saves results in `result_ini.txt` within the `result_ini` directory.
- **Frame Limitation:** The script processes the entire video by default. Uncomment and modify the `while` loop condition to limit processing to the first 20 frames.

### Output Files

- **Annotated Video:** The script will generate an output video file `output_video.avi` with detected objects and their tracking IDs.
- **Tracking Data:** A text file `result_ini.txt` will store the tracking data in the format:
  ```
  track_id class_id x y frame_count
  ```
  where `x` and `y` are the coordinates of the tracked object's center.

## Key Components

### SORT Tracker

The Simple Online and Realtime Tracking (SORT) algorithm is used to track the detected objects frame-by-frame. The tracks are stored and updated using Kalman filtering to estimate the object's future positions.

### Detectron2

Detectron2 is utilized for object detection. This implementation uses the Mask R-CNN model with a ResNet-50 backbone. The configuration can be modified to use different models or settings as needed.

### Visualizer

The visualizer is responsible for rendering the detection and tracking information on each video frame. Annotations such as bounding boxes and tracking IDs are drawn on the frame before saving.

## Notes

- **CPU Usage:** The script is configured to run on the CPU by default. Modify `cfg.MODEL.DEVICE` to `"cuda"` if a GPU is available and you wish to use it.
- **Customization:** You can easily customize the model, dataset, and parameters in the configuration section of the script to adapt the tracking system to different scenarios or datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Detectron2** by Facebook AI Research (FAIR) for providing the object detection framework.
- **SORT** by Alex Bewley for the object tracking algorithm.

Feel free to modify and adapt this repository to suit your needs. Contributions are welcome!
