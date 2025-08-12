import cv2
import numpy as np
import threading
import time
import os
import re

# 从独立的工具文件导入串口通信和日志功能
from serial_utils import find_and_open_serial_ports, serial_lock, log_to_screen, send_command, ser, ser_voltage

# 导入功能模块，模型将在导入时预加载
from single_shape import detect_shape_and_size
from multiple_squares import detect_multiple_squares
from digit_squares import detect_digit_squares
from overlapping_squares import detect_overlapping_squares

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

BORDER_WIDTH_CM = 2.0

# --- 分段焦距常量 ---
FOCAL_LENGTH_SHORT_DISTANCE = 16619.48  # 测量距离小于125cm时使用
FOCAL_LENGTH_LONG_DISTANCE = 16443.17   # 测量距离大于125cm时使用
DISTANCE_THRESHOLD = 125.0              # 距离阈值

# --- 全局变量和模式映射 ---
current_mode = 0
target_digit = -1

# 新增全局变量和常量
max_power_val = 0.0
VOLTAGE_BIAS = 0.01
VOLTAGE_FIXED = 5.0 # 5V固定电压
VOLTAGE_UPDATE_INTERVAL = 1.0 # 1秒刷新一次

MODE_SINGLE_SHAPE = 1
MODE_MULTIPLE_SQUARES = 2
MODE_DIGIT_SQUARES = 3
MODE_OVERLAPPING_SQUARES = 4

# 按键ID映射表 (根据删除p0(id=1)后，所有ID-1的最终版)
BUTTON_MAP = {
    3: "NUM_1",
    4: "NUM_2",
    5: "NUM_3",
    6: "NUM_4",
    7: "NUM_5",
    8: "NUM_6",
    9: "NUM_7",
    10: "NUM_8",
    11: "NUM_9",
    12: "NUM_0",
    13: "EXIT",
    19: "MODE_A",
    20: "MODE_B",
    21: "MODE_C",
    22: "MODE_D" # 映射到功能4
}

# --- 线程一：按键事件处理 ---
def handle_button_press(button_name):
    """根据按钮名称执行不同操作"""
    global current_mode, target_digit
    log_to_screen(f"处理按键: {button_name}")
    send_command(f't0.txt="按键: {button_name}"')
    
    if button_name == "MODE_A":
        current_mode = MODE_SINGLE_SHAPE
        send_command('t_status.txt="模式1: 单个图形"')
        log_to_screen("已切换到功能1：检测单个几何图形")
    elif button_name == "MODE_B":
        current_mode = MODE_MULTIPLE_SQUARES
        send_command('t_status.txt="模式2: 多个正方形"')
        log_to_screen("已切换到功能2：检测多个正方形")
    elif button_name == "MODE_C":
        current_mode = MODE_DIGIT_SQUARES
        target_digit = -1 # 等待数字输入
        send_command('t_status.txt="模式3: 数字识别"')
        send_command('t_result1.txt="请按数字键(0-9)"')
        log_to_screen("已切换到功能3：请等待数字按键输入...")
    elif button_name == "MODE_D":
        current_mode = MODE_OVERLAPPING_SQUARES
        send_command('t_status.txt="模式4: 重叠正方形"')
        log_to_screen("已切换到功能4：检测重叠的最小面积正方形")
    elif "NUM_" in button_name and current_mode == MODE_DIGIT_SQUARES:
        target_digit = int(button_name.split('_')[1])
        send_command(f't_result1.txt="目标数字: {target_digit}"')
        log_to_screen(f"已设置目标数字: {target_digit}")
    elif button_name == "EXIT":
        current_mode = 0
        target_digit = -1
        send_command('t_status.txt="等待选择模式..."')
        send_command('t_result1.txt=""')
        send_command('t_result2.txt=""')
        send_command('t_result3.txt=""')
        log_to_screen("已退出当前功能，等待下一次选择。")

def serial_listener_thread():
    """后台线程：专门监听来自屏幕的按键消息"""
    log_to_screen("串口监听线程已启动...")
    # 全局ser在main函数中已赋值
    if not ser or not ser.is_open:
        log_to_screen("串口未打开，监听线程退出。")
        return

    while True:
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                for i in range(0, len(data), 7):
                    chunk = data[i:i + 7]
                    if len(chunk) == 7 and chunk[0] == 0x65 and chunk[3] == 0x01:
                        component_id = chunk[2]
                        if component_id in BUTTON_MAP:
                            handle_button_press(BUTTON_MAP[component_id])
            time.sleep(0.05)
        except Exception as e:
            log_to_screen(f"串口监听出错: {e}")
            break

# --- 新增线程二：电压数据处理 ---
def voltage_listener_thread():
    """后台线程：监听来自电压模块的串口数据，并计算功耗"""
    global max_power_val
    log_to_screen("电压监听线程已启动...")
    # 全局ser_voltage在main函数中已赋值
    if not ser_voltage or not ser_voltage.is_open:
        log_to_screen("电压串口未打开，监听线程退出。")
        return

    last_voltage_update_time = time.time()

    while True:
        try:
            line = ser_voltage.readline().decode('utf-8')
            if "Voltage:" in line:
                match = re.search(r"Voltage: (\d+\.\d+) V", line)
                if match:
                    voltage_val = float(match.group(1))
                    
                    # 计算电流和功率
                    current_val = voltage_val / 50 / (0.02 + VOLTAGE_BIAS)
                    power_val = current_val * VOLTAGE_FIXED
                    
                    # 更新最大功耗
                    if power_val > max_power_val:
                        max_power_val = power_val
                    
                    # 降低发送频率，每1秒更新一次屏幕
                    if time.time() - last_voltage_update_time >= VOLTAGE_UPDATE_INTERVAL:
                        # 发送数据到串口屏
                        send_command(f't_cur.txt="Current:{current_val:.3f}A"')
                        send_command(f't_pow.txt="Power:{power_val:.3f}W"')
                        send_command(f't_maxpower.txt="最大功耗:{max_power_val:.3f}W"')
                        
                        log_to_screen(f"电压: {voltage_val:.3f}V, 电流: {current_val:.3f}A, 功率: {power_val:.3f}W")
                        
                        last_voltage_update_time = time.time()
            
            time.sleep(0.01) # 短暂休眠以避免CPU占用过高
        except Exception as e:
            log_to_screen(f"电压串口监听出错: {e}")
            break

def find_largest_rectangle(contours):
    max_area = 0
    best_approx = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_approx = approx
    return best_approx

def calculate_distance(focal_length, real_width, pixel_width):
    if pixel_width == 0:
        return 0
    return (real_width * focal_length) / pixel_width

def main():
    global current_mode, target_digit, ser, ser_voltage
    
    # 自动查找并打开串口，并赋值给全局变量
    ser, ser_voltage = find_and_open_serial_ports()

    if ser and ser.is_open:
        listener = threading.Thread(target=serial_listener_thread, daemon=True)
        listener.start()
        send_command('t_status.txt="等待选择模式..."')
    else:
        log_to_screen("警告: 未找到串口屏，无法进行交互。")
    
    if ser_voltage and ser_voltage.is_open:
        voltage_listener = threading.Thread(target=voltage_listener_thread, daemon=True)
        voltage_listener.start()
    else:
        log_to_screen("警告: 未找到电压模块，无法接收电压数据。")

    if not ser and not ser_voltage:
        log_to_screen("两个串口均未打开，程序将退出。")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        log_to_screen("错误: 无法打开摄像头。请检查CAMERA_INDEX。")
        return
    log_to_screen("摄像头已启动。")

    log_to_screen("程序已启动，等待串口屏指令。")
    
    last_update_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            log_to_screen("错误: 无法读取摄像头帧。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = find_largest_rectangle(contours)

        distance = 0
        if rect is not None:
            pts = rect.reshape(4, 2)
            edge1 = np.linalg.norm(pts[0] - pts[1])
            edge2 = np.linalg.norm(pts[1] - pts[2])
            pixel_width = max(edge1, edge2)
            
            # --- 分段焦距逻辑 ---
            # 这里的距离计算是一个初步估计，用于选择焦距
            # 正确的距离计算应该在选择焦距后进行
            focal_length_to_use = FOCAL_LENGTH_SHORT_DISTANCE
            initial_distance = calculate_distance(FOCAL_LENGTH_SHORT_DISTANCE, BORDER_WIDTH_CM, pixel_width)
            if initial_distance > DISTANCE_THRESHOLD:
                focal_length_to_use = FOCAL_LENGTH_LONG_DISTANCE
            
            distance = calculate_distance(focal_length_to_use, BORDER_WIDTH_CM, pixel_width)

            # 透视变换以矫正A4纸
            def order_points(pts):
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                return rect
            rect_pts = order_points(pts.astype("float32"))
            dst_width = 420
            dst_height = int(dst_width * 1.414)
            dst_pts = np.array([[0, 0], [dst_width, 0], 
                               [dst_width, dst_height], [0, dst_height]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            warp = cv2.warpPerspective(frame, M, (dst_width, dst_height))
        else:
            warp = None
            log_to_screen("未检测到A4纸边界。")
            send_command('t_result1.txt="未找到A4纸"')

        # --- 检查是否需要更新屏幕结果 ---
        if time.time() - last_update_time > 0.5:
            if warp is not None:
                if current_mode == MODE_SINGLE_SHAPE:
                    log_to_screen("\n执行功能1: 单个几何图形检测")
                    detect_shape_and_size(warp, focal_length_to_use, distance)
                elif current_mode == MODE_MULTIPLE_SQUARES:
                    log_to_screen("\n执行功能2: 多个正方形检测")
                    detect_multiple_squares(warp, focal_length_to_use, distance)
                elif current_mode == MODE_DIGIT_SQUARES and target_digit != -1:
                    log_to_screen(f"\n执行功能3: 目标数字 {target_digit} 识别")
                    detect_digit_squares(warp, target_digit, focal_length_to_use, distance)
                elif current_mode == MODE_OVERLAPPING_SQUARES:
                    log_to_screen("\n执行功能4: 重叠正方形检测")
                    detect_overlapping_squares(warp, focal_length_to_use, distance)
            
            last_update_time = time.time()
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    if ser and ser.is_open:
        ser.close()
    if ser_voltage and ser_voltage.is_open:
        ser_voltage.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()