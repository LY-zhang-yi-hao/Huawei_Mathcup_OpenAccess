from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
#
model = YOLO("yolov8n.pt")

OBJ_LIST = ["person", "car", "bus", "truck"]
VIDEO_PATH = r"D:\Desktop\JS_jiemi\高速公路交通流数据\32.31.250.103\20240501_20240501125647_20240501140806_125649.mp4"
RESULT_PATH = "result2.mp4"

# 记录所有id的位置信息
track_history = defaultdict(lambda: [])

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()
    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取宽度
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取高度

    videoWriter = None

    while True:
        success, frame = capture.read()  # 读取视频中的一帧
        if not success:
            print("读取帧失败")
            break

        results = model.track(frame, persist=True)

        # 可视化显示目标检测框
        a_frame = results[0].plot()
        # 所有id的位置信息
        boxes = results[0].boxes.xywh.cpu()
        # 所有ID的序列号信息
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist()
        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            # 获取当前类别名
            class_name = results[0].names[cls_id]

            # 最长为50个点
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 50:
                track.pop(0)
            # 当前的track_id所有经过的轨迹路径点（不超过50个）
            points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)

            cv2.polylines(a_frame, [points], isClosed=False, color=(255, 0, 255), thickness=3)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (int(frame_width), int(frame_height)))

        videoWriter.write(a_frame)
        cv2.imshow("yolo track", a_frame)
        cv2.waitKey(1)

    capture.release()  # 释放capture资源
    videoWriter.release()
    cv2.destroyAllWindows()  # 关闭掉所有的窗口