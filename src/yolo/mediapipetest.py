import cv2
import mediapipe as mp

# 初始化 MediaPipe 的绘图和手部模块
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 创建手部检测对象
hands = mp_hands.Hands(
    static_image_mode=False,       # 设置为 False 表示视频流输入
    max_num_hands=2,               # 最多检测两只手
    min_detection_confidence=0.75, # 置信度阈值
    min_tracking_confidence=0.75   # 跟踪阈值
)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将 BGR 图像转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像，检测手部关键点
    results = hands.process(frame_rgb)

    # 如果检测到手
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 在图像上绘制手部关键点和连接线
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示处理后的图像
    cv2.imshow('MediaPipe Hands', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
