from pyorbbecsdk import *
import cv2
import numpy as np
import yaml
import time
import os
import sys
from utils import frame_to_bgr_image

ESC_KEY = 27
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm
font = cv2.FONT_HERSHEY_SIMPLEX

# 相机光心坐标：
cx = 644.752
cy = 395.306
# 焦距
fx = 611.963
fy = 611.742

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global last_click_info
        color_image, depth_data, scale = param
        z = depth_data[y, x] * scale
        if z == 0:
            print(f"深度值无效，无法计算实际坐标")
            return
        x_mm = (x - cx) * z / fx  # x方向向右
        y_mm = (y - cy) * z / fy  # y方向向下
        print(f"鼠标点击坐标: ({x}, {y})")
        print(f"实际坐标: ({x_mm:.2f} mm, {y_mm:.2f} mm, {z:.2f} mm)")
        
        # 保存点击信息
        last_click_info = (x, y, x_mm, y_mm, z)

def main(argv):
    global last_click_info
    last_click_info = None

    # 创建管道
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)
    config = Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1280, 800, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_video_stream_profile(1280, 800, OBFormat.Y16, 30)
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return

    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return

    cv2.namedWindow("frame")

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(10)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            depth_data = temporal_filter.process(depth_data)

            # 将color_image和depth_data传递给mouse_callback
            cv2.setMouseCallback("frame", mouse_callback, param=(color_image, depth_data, scale))

            # 如果有点击信息，显示在图像上
            if last_click_info is not None:
                x, y, x_mm, y_mm, z = last_click_info
                text = f"({x_mm:.2f}, {y_mm:.2f}, {z:.2f}) mm"
                cv2.putText(color_image, text, (x, y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)  # 在点击处画一个红色圆点

            cv2.imshow("frame", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
            elif key == ord(' '):  # 按空格键保存图片
                filename = str(time.time())[:10] + ".jpg"
                save_dir = "pic"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                full_path = os.path.join(save_dir, filename)
                cv2.imwrite(full_path, color_image)
        except KeyboardInterrupt:
            break

    pipeline.stop()

if __name__ == "__main__":
    main(sys.argv[1:])