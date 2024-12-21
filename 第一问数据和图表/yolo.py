import cv2  # 导入OpenCV库，用于处理视频和图像
import torch  # 导入PyTorch库，用于加载和使用深度学习模型
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和保存


# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 使用PyTorch的hub功能加载YOLOv5模型，'yolov5s'表示加载的是YOLOv5的小型版本
def get_video_info(video_path):  # 定义一个函数，用于获取视频的基本信息
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    if not cap.isOpened():  # 检查视频是否成功打开
        print("无法打开视频文件")
        return None, None, None  # 如果无法打开视频，返回None

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率（每秒帧数）
    # 获取视频时长（秒）
    duration = total_frames / fps  # 计算视频的时长（秒）
    print(f"视频总帧数: {total_frames}")  # 打印视频总帧数
    print(f"视频帧率: {fps}")  # 打印视频帧率
    print(f"视频时长（秒）: {duration}")  # 打印视频时长
    cap.release()  # 释放视频文件
    return total_frames, fps, duration  # 返回视频的总帧数、帧率和时长

def process_video(video_path, output_excel="traffic_data.xlsx", vehicle_length=4.0, safe_gap=2.0, frame_skip=10):  # 定义一个函数，用于处理视频并计算交通数据

    # 获取视频信息
    total_frames, fps, duration = get_video_info(video_path)  # 调用get_video_info函数获取视频信息
    if total_frames is None or fps is None or duration is None:  # 检查视频信息是否获取成功
        print("视频信息获取失败")
        return  # 如果获取失败，返回

    cap = cv2.VideoCapture(video_path)  # 重新打开视频文件
    if not cap.isOpened():  # 检查视频是否成功打开
        print("无法打开视频文件")
        return  # 如果无法打开视频，返回

    frame_count = 0  # 初始化帧计数器
    vehicle_counts = []  # 初始化车辆数量列表

    while True:  # 开始循环读取视频帧
        ret, frame = cap.read()  # 读取一帧视频
        if not ret:  # 检查是否成功读取帧
            break  # 如果读取失败，跳出循环
        # 仅处理每 frame_skip 帧中的一帧

        if frame_count % frame_skip != 0:  # 检查当前帧是否需要跳过
            frame_count += 1  # 增加帧计数器
            continue  # 跳过当前帧

        frame_count += 1  # 增加帧计数器

        # 使用YOLO模型进行车辆检测
        results = model(frame)  # 使用YOLOv5模型对当前帧进行车辆检测
        # 过滤检测结果，只保留车辆相关类别
        vehicles = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'].isin(['car', 'truck', 'bus', 'motorbike'])]  # 过滤检测结果，只保留车辆类别（汽车、卡车、公交车、摩托车）
        # 统计当前帧的车辆数量

        vehicle_count = len(vehicles)  # 计算当前帧的车辆数量
        vehicle_counts.append(vehicle_count)  # 将车辆数量添加到列表中

        # 清理内存
        vehicles = None  # 释放vehicles变量占用的内存
        results = None  # 释放results变量占用的内存

    cap.release()  # 释放视频文件

    # 计算交通密度
    avg_vehicle_count = np.mean(vehicle_counts)  # 计算平均车辆数量
    road_length_estimate = avg_vehicle_count * (vehicle_length + safe_gap)  # 估算道路长度
    density = avg_vehicle_count / road_length_estimate  # 计算交通密度
    # 计算交通流量
    duration_per_frame = frame_skip / fps  # 每个采样的实际时间
    flow_rate = avg_vehicle_count / duration_per_frame  # 计算交通流量

    print(f"平均车辆数量: {avg_vehicle_count}")  # 打印平均车辆数量
    print(f"估算道路长度: {road_length_estimate:.2f} 米")  # 打印估算道路长度
    print(f"交通密度: {density:.4f} 辆/米")  # 打印交通密度
    print(f"交通流量: {flow_rate:.2f} 辆/秒")  # 打印交通流量
    # 保存结果到Excel

    data = {

        "车辆数量": [avg_vehicle_count],  # 车辆数量
        "估算道路长度 (米)": [road_length_estimate],  # 估算道路长度
        "交通密度 (辆/米)": [density],  # 交通密度
        "交通流量 (辆/秒)": [flow_rate]  # 交通流量

    }
    df = pd.DataFrame(data)  # 将数据转换为DataFrame
    df.to_excel(output_excel, index=False)  # 将数据保存到Excel文件
    print(f"数据已保存到 {output_excel}")  # 打印保存路径

# 示例：使用视频文件进行处理并保存结果到Excel

video_path = r"D:\Desktop\JS_jiemi\高速公路交通流数据\32.31.250.103\20240501_20240501125647_20240501140806_125649.mp4"  # 替换为你的视频路径
output_excel = r"D:\Desktop\JS_jiemi\高速公路交通流数据\32.31.250.103/traffic_data.xlsx"  # 替换为你的输出Excel路径
process_video(video_path, output_excel)  # 调用process_video函数处理视频并保存结果