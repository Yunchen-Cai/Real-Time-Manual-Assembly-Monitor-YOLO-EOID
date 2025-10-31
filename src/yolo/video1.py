import json


def generate_frame_labels(video_timeline, fps=25, output_file="frame_labels.json"):
    # 计算总帧数
    total_duration = max(segment["end"] for segment in video_timeline)
    total_frames = int(total_duration * fps) + 1

    # 初始化帧标签列表
    frame_labels = []

    # 遍历每一帧
    for frame_idx in range(total_frames):
        time_stamp = frame_idx / fps  # 当前帧的时间戳

        # 匹配标签（增加浮点容差）
        current_action = None
        current_object = None
        for segment in video_timeline:
            if (time_stamp + 1e-6) >= segment["start"] and (time_stamp - 1e-6) < segment["end"]:
                current_action = segment["action"]
                current_object = segment["object"]
                break

        # 记录帧标签
        frame_labels.append({
            "frame_index": frame_idx,
            "time_stamp": round(time_stamp, 3),
            "action": current_action,
            "object": current_object
        })

    # 保存为JSON文件
    with open(output_file, "w") as f:
        json.dump(frame_labels, f, indent=2)

    print(f"已生成 {output_file}，共 {total_frames} 帧。")


# 示例调用（适配实际路径）
input_path = r"C:\Users\32894\Desktop\data\frameano\video6_timeline.json"
output_path = r"C:\Users\32894\Desktop\data\frameano\video6_frame_labels.json"

with open(input_path, "r") as f:
    video1_timeline = json.load(f)
generate_frame_labels(video1_timeline, output_file=output_path)