import cv2
import os

# 设定路径
image_folder = r"C:\Users\32894\Desktop\data\1741676273625"
output_video = r"C:\Users\32894\Desktop\output\assembled_video.mp4"
frame_rate = 5

# 获取所有图片文件，并按时间戳排序
images = sorted(
    [img for img in os.listdir(image_folder) if img.endswith(".png")],
    key=lambda x: int(x.split('.')[0])  # 按时间戳排序
)

# 确保输出目录存在
output_dir = os.path.dirname(output_video)
os.makedirs(output_dir, exist_ok=True)

# 读取第一张可用图片确定分辨率
first_frame = None
for img in images:
    first_frame = cv2.imread(os.path.join(image_folder, img))
    if first_frame is not None:
        break  # 找到第一张有效图片，跳出循环

if first_frame is None:
    print("❌ 所有图片都无法读取，视频生成失败！")
    exit()

h, w, _ = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (w, h))

# 逐帧写入视频
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    if frame is None:
        print(f"⚠️ 跳过损坏的图片: {image}")
        continue
    video.write(frame)

video.release()
cv2.destroyAllWindows()
print(f"✅ 视频已保存到: {output_video}")
