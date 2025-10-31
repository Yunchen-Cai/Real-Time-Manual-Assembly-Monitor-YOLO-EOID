from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import threading
import queue
from test_on_images import run_on_images, load_model, get_args_parser, transfer_image
import os
import time

app = Flask(__name__)

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化队列和锁
MAX_QUEUE_SIZE = 10000
image_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)  # 最大缓冲区大小
processed_images = queue.Queue()
queue_lock = threading.Lock()  # 添加线程锁，确保顺序处理

countV = 1
def count():
    global countV
    countV += 1
    return countV

def process_image_from_queue():
    """后台线程，用来从队列中获取图片并进行处理"""
    while True:
        queue_lock.acquire()  # 获取锁，保证线程安全
        if not image_queue.empty():
            img = image_queue.get()  # 按顺序取出最早的一张图片
            queue_lock.release()  # 释放锁
        else:
            queue_lock.release()  # 释放锁
            time.sleep(0.05)  # 适当休眠，防止 CPU 过载
            continue  # 继续下一轮循环

        # 实时打印队列占用率
        # queue_size = image_queue.qsize()
        # queue_usage = (queue_size / MAX_QUEUE_SIZE) * 100
        # print(f"Queue usage: {queue_size}/{MAX_QUEUE_SIZE} ({queue_usage:.2f}%)")


        print(f"Processing image...")  # 可选调试信息
        result = transfer_image(img)
        hoi_list, img_result = result['hoi_list'], result['img_result']

        # 确保 hoi_list 非空再处理
        if hoi_list:
            i_cls, i_name = hoi_list[0]['i_cls'], hoi_list[0]['i_name']
            action = [{"actionAndObject": i_name, "probability": i_cls}]
            processed_images.put({"actionObservations": action, "t": time.time()})
            print(f"actionAndObject: {i_name}")
        else:
            print("No valid predictions found, skipping image processing.")
            processed_images.put({"error": "No valid predictions"})

# 启动一个处理图片的线程
processing_thread = threading.Thread(target=process_image_from_queue)
processing_thread.daemon = True  # 设置为守护线程，主程序退出时自动退出
processing_thread.start()

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # 检查队列是否已满，如果满了，返回busy状态
        if image_queue.full():
            return jsonify({"status": "busy", "message": "Server is processing too many images. Please try again later."}), 503
        # 从请求中获取图片数据
        image_data = request.data
        # 转换为OpenCV格式
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 可选：进行额外处理，例如resize、灰度化等
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 将接收到的图片放入队列等待处理
        image_queue.put(img)

        # 在这里打印队列大小，确保图片入队
        queue_lock.acquire()
        queue_size = image_queue.qsize()
        queue_lock.release()
        queue_usage = (queue_size / MAX_QUEUE_SIZE) * 100
        print(f"Queue usage: {queue_size}/{MAX_QUEUE_SIZE} ({queue_usage:.2f}%)")

        # 等待图片处理完毕并返回结果
        processed_result = processed_images.get()

        # 返回处理结果
        return jsonify(processed_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
