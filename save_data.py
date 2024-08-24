import os, json


import cv2


# 保存识别的图片和数据
def save_data(id, class_name, x1, y1, x2, y2, img, frame_count):
    result_path = "initial_result"
    save_dir = os.path.join(result_path, str(id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_image_dir = os.path.join(save_dir, "images")
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    save_path = os.path.join(save_image_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(save_path, img)

    # 收集要保存的数据
    data = {
        "Frame": frame_count,
        "ID": int(id),
        "Category": class_name,
        "Box": [int(x1), int(y1), int(x2), int(y2)],
    }
    json_file_path = os.path.join(save_dir, "initial_data.json")
    if os.path.exists(json_file_path):
        # 如果文件已经存在，去掉最后一个字符（即 ']'），然后添加新的数据
        with open(json_file_path, "r+") as file:
            file.seek(0, os.SEEK_END)
            file.seek(file.tell() - 1, os.SEEK_SET)
            file.truncate()
            file.write(",\n")
            json.dump(data, file)
            file.write("]")  # 重新加上结束括号
    else:
        # 如果文件不存在，创建一个新的 JSON 数组文件
        with open(json_file_path, "w") as file:
            file.write("[\n")
            json.dump(data, file)
            file.write("\n]")
