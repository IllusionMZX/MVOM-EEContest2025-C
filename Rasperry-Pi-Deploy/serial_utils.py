import serial
import threading
import time
import glob
import re

# --- 串口配置 ---
BAUD_RATE = 115200

# 声明全局变量，但在此处不初始化
ser = None
ser_voltage = None
serial_lock = threading.Lock()

def send_command(cmd_str):
    """线程安全的发送函数，发送普通文本指令到屏幕"""
    if ser and ser.is_open:
        with serial_lock:
            try:
                time.sleep(0.02)
                ser.write(cmd_str.encode('gb2312'))
                ser.write(b'\xff\xff\xff')
            except Exception as e:
                print(f"发送串口指令出错: {e}")

# --- 全局日志缓冲区和函数 ---
MAX_LOG_LINES = 5
log_buffer = []

def log_to_screen(message):
    """将日志信息打印到终端，并更新到屏幕的t_log文本框"""
    print(message)
    log_buffer.append(message)
    if len(log_buffer) > MAX_LOG_LINES:
        log_buffer.pop(0)
    screen_text = "\\r".join(log_buffer)
    send_command(f't_log.txt="{screen_text}"')

def find_and_open_serial_ports():
    """
    自动查找并打开显示屏和电压模块的串口。
    通过向串口发送指令或读取特定格式的数据来识别设备。
    """
    global ser, ser_voltage
    
    ports = glob.glob('/dev/ttyUSB*')
    print(f"找到可用的串口设备: {ports}")

    # 优先尝试识别电压模块，因为其数据格式是可预测的
    for port in ports:
        if ser_voltage:
            break
        print(f"正在尝试打开并识别电压模块: {port}")
        temp_ser = None
        try:
            temp_ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            time.sleep(1) # 等待设备初始化

            # 读取原始字节，并尝试解码
            raw_bytes = temp_ser.read(temp_ser.in_waiting or 1)
            try:
                line = raw_bytes.decode('utf-8')
                if "Voltage:" in line:
                    print(f"已识别电压模块在: {port}")
                    ser_voltage = temp_ser
                    break # 找到并跳出循环
                else:
                    temp_ser.close()
            except UnicodeDecodeError:
                # 忽略解码错误，继续尝试下一个端口
                print(f"端口 {port} 无法以 UTF-8 解码，可能不是电压模块。")
                temp_ser.close()
            
        except serial.SerialException as e:
            print(f"打开或测试串口 {port} 失败: {e}")
            if temp_ser and temp_ser.is_open:
                temp_ser.close()

    # 识别显示屏
    for port in ports:
        # 如果这个端口已经被识别为电压模块，则跳过
        if ser_voltage and port == ser_voltage.port:
            continue

        print(f"正在尝试打开并识别串口屏: {port}")
        temp_ser = None
        try:
            temp_ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            time.sleep(1) # 等待设备初始化

            # 尝试识别串口屏: 发送一个指令，如果成功没有异常，则假设为串口屏
            test_cmd_str = 't0.txt="Test"'
            temp_ser.write(test_cmd_str.encode('gb2312'))
            temp_ser.write(b'\xff\xff\xff') # 发送原始字节，避免编码错误
            time.sleep(0.1)
            
            # 如果没有异常，就认为是串口屏
            print(f"已识别串口屏在: {port}")
            ser = temp_ser
            break # 找到并跳出循环
        except serial.SerialException as e:
            print(f"打开或测试串口 {port} 失败: {e}")
            if temp_ser and temp_ser.is_open:
                temp_ser.close()
        except Exception as e:
            print(f"测试串口 {port} 时发生意外错误: {e}")
            if temp_ser and temp_ser.is_open:
                temp_ser.close()
    
    if not ser:
        print("警告: 未能找到串口屏设备。")
    if not ser_voltage:
        print("警告: 未能找到电压模块设备。")
    
    return ser, ser_voltage