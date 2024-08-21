import re


def shift_time(time_str, offset):
    """Shift the given time string by the offset in seconds."""
    time_parts = re.split(r"[:.]", time_str)
    h, m = map(int, time_parts[:2])
    s = float(f"{time_parts[2]}.{time_parts[3]}")
    total_seconds = h * 3600 + m * 60 + s + offset
    if total_seconds < 0:
        total_seconds = 0
    new_h = int(total_seconds // 3600)
    new_m = int((total_seconds % 3600) // 60)
    new_s = total_seconds % 60
    return f"{new_h:01}:{new_m:02}:{new_s:05.2f}"


def adjust_ass_timing(file_path, offset):
    """Adjust the timing in an .ass file by a given offset in seconds."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            if line.startswith("Dialogue:"):
                parts = line.split(",")
                start_time = parts[1].replace(",", ".")
                end_time = parts[2].replace(",", ".")
                new_start = shift_time(start_time, offset)
                new_end = shift_time(end_time, offset)
                parts[1] = new_start
                parts[2] = new_end
                file.write(",".join(parts))
            else:
                file.write(line)


# 使用方法
file_path = "简英双字.ass"  # 替换为你的 .ass 文件路径
offset = -30  # 提前 30 秒
adjust_ass_timing(file_path, offset)
