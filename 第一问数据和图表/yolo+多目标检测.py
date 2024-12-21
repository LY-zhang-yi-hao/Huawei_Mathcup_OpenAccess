import cv2
import torch
from sort import Sort  # 导入SORT跟踪算法
import numpy as np

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# SORT跟踪器实例化
tracker = Sort()

# 用于存储每辆车的速度
vehicle_speeds = {}
# 用于存储每辆车的历史位置
vehicle_positions = {}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率

    # 调整像素与实际距离的比例，假设视频中1000像素代表实际75米
    pixel_distance = 1500  # 假设视频中的一段路长是1500像素
    actual_distance = 100  # 实际路段长度为100米
    scale_factor = actual_distance / pixel_distance  # 每个像素代表多少米

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 使用YOLOv5进行车辆检测
        results = model(frame)

        # 过滤检测结果，只保留车辆相关类别
        vehicles = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'].isin(['car', 'truck', 'bus', 'motorbike'])]

        # 将YOLO检测结果转换为[左上角X, 左上角Y, 右下角X, 右下角Y, 置信度]
        detections = vehicles[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy()

        # 使用SORT进行车辆跟踪
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]  # 获取跟踪的边界框和跟踪ID
            # 计算车辆的中心点位置
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 如果车辆第一次出现，初始化它的位置信息
            if track_id not in vehicle_positions:
                vehicle_positions[track_id] = {'previous_position': (center_x, center_y), 'frame_count': frame_count}
            else:
                # 计算当前帧与上一帧之间的时间差
                prev_x, prev_y = vehicle_positions[track_id]['previous_position']
                previous_frame = vehicle_positions[track_id]['frame_count']

                # 计算像素位移
                distance_pixels = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                # 将像素位移转换为实际位移（米）
                real_distance = distance_pixels * scale_factor
                # 计算经过的时间（秒）
                time_elapsed = (frame_count - previous_frame) / fps

                if time_elapsed > 0:
                    # 计算速度（米/秒），并存储到 vehicle_speeds 字典中
                    speed_mps = real_distance / time_elapsed
                    speed_kmh = speed_mps * 3.6  # 转换为公里/小时

                    if track_id not in vehicle_speeds:
                        vehicle_speeds[track_id] = []

                    vehicle_speeds[track_id].append(speed_kmh)

                # 更新车辆位置和帧数
                vehicle_positions[track_id]['previous_position'] = (center_x, center_y)
                vehicle_positions[track_id]['frame_count'] = frame_count

            # 可视化跟踪框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2)

        # 显示处理后的帧
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 计算所有车辆的平均速度，剔除速度超过75 km/h的车辆
    calculate_average_speed(vehicle_speeds)

def calculate_average_speed(vehicle_speeds):
    total_speed = 0
    total_cars = 0

    # 计算每辆车的平均速度并剔除速度超过75 km/h的车辆
    for track_id, speeds in vehicle_speeds.items():
        if len(speeds) > 0:
            # 剔除速度大于75 km/h的数据
            filtered_speeds = [speed for speed in speeds if speed <= 75]
            if len(filtered_speeds) > 0:
                avg_speed = np.mean(filtered_speeds)
                total_speed += avg_speed
                total_cars += 1
                print(f'Vehicle ID: {int(track_id)}, Average Speed: {avg_speed:.2f} km/h')

    # 计算视频中所有车辆的平均速度
    if total_cars > 0:
        overall_average_speed = total_speed / total_cars
        print(f'Overall Average Speed of All Vehicles: {overall_average_speed:.2f} km/h')
    else:
        print("No vehicles tracked below 75 km/h.")


# 视频路径
video_path = r"D:\Desktop\JS_jiemi\高速公路交通流数据\32.31.250.103\20240501_20240501125647_20240501140806_125649.mp4"
process_video(video_path)
