import cv2
import mediapipe as mp

# 初始化 MediaPipe 的手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 加载视频文件
cap = cv2.VideoCapture(r'C:\Users\32894\Desktop\xiang.mp4')  # 替换为你的视频路径

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置视频编码和输出文件
out = cv2.VideoWriter(r'C:\Users\32894\Desktop\象印.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                      (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将 BGR 转换为 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理视频帧，检测手部
    result = hands.process(image_rgb)
    
    # 如果检测到手部，绘制关键点
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 将处理后的帧写入输出文件
    out.write(frame)

    # 显示带有关键点的帧
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:  # 按 ESC 键退出
        break

# 释放资源
cap.release()
out.release()  # 保存并关闭输出视频
cv2.destroyAllWindows()
