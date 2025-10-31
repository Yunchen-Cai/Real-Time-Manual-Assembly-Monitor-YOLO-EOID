import cv2
import mediapipe as mp

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 读取输入图片
image_path = "E:\\ONGSKFYP\\add_frames\\train\\frame_0336.png"  # 替换成你的图片路径
output_path = "E:\\ONGSKFYP\\add_frames\\output.png"

image = cv2.imread(image_path)

# 将 BGR 图像转换为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 处理图像，检测手部关键点
with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    results = hands.process(image_rgb)

    # 在图像上绘制关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# 保存处理后的图片
cv2.imwrite(output_path, image)
print(f"处理后的图片已保存: {output_path}")

# 显示图像
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
