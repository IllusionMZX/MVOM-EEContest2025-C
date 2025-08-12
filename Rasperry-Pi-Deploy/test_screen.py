import serial
import time
import threading
import random

# --- 串口配置 ---
SERIAL_PORT = '/dev/ttyUSB1'  # 在Windows上。如果在树莓派上，请使用 '/dev/serial0' 或 '/dev/ttyUSB0'
BAUD_RATE = 115200  # 使用一个稳定可靠的波特率

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except serial.SerialException as e:
    print(f"打开串口失败: {e}")
    exit()

# --- 线程锁，确保串口访问安全 ---
serial_lock = threading.Lock()

# --- 用于屏幕日志的全局变量 ---
log_buffer = []
MAX_LOG_LINES = 10  # 屏幕上最多显示10行日志

# --- 按键ID映射表 (根据删除p0(id=1)后，所有ID-1的最终版) ---
BUTTON_MAP = {
    3: "NUM_1",  # 原ID 4
    4: "NUM_2",  # 原ID 5
    5: "NUM_3",  # 原ID 6
    6: "NUM_4",  # 原ID 7
    7: "NUM_5",  # 原ID 8
    8: "NUM_6",  # 原ID 9
    9: "NUM_7",  # 原ID 10
    10: "NUM_8",  # 原ID 11
    11: "NUM_9",  # 原ID 12
    12: "NUM_0",  # 原ID 13
    13: "EXIT",  # 原ID 14
    19: "MODE_A",  # 原ID 20
    20: "MODE_B",  # 原ID 21
    21: "MODE_C",  # 原ID 22
    22: "MODE_D",  # 原ID 23
    23: "MODE_E",  # 原ID 24
    24: "MODE_F"  # 原ID 25
}


# --- 超级Print函数，同时输出到终端和屏幕 ---
def log_to_screen(message):
    """将日志信息打印到终端，并更新到屏幕的t_log文本框"""
    # 1. 在本地终端打印，方便调试
    print(message)

    # 2. 更新日志缓冲区
    log_buffer.append(message)
    # 如果日志超过最大行数，就移除最老的一行
    if len(log_buffer) > MAX_LOG_LINES:
        log_buffer.pop(0)

    # 3. 格式化准备发送到屏幕的文本
    # 串口屏的换行符是 \r
    screen_text = "\\r".join(log_buffer)

    # 4. 发送指令更新t_log文本框
    send_command(f't_log.txt="{screen_text}"')


# --- 线程一：按键事件处理 ---
def handle_button_press(button_name):
    """根据按钮名称执行不同操作"""
    # 使用新的日志函数来输出信息
    log_to_screen(f"处理按键: {button_name}")

    if "MODE_" in button_name:
        mode = button_name.split('_')[1]
        log_to_screen(f"启动模式 {mode} 的识别程序...")
        send_command(f't0.txt="模式 {mode} 已激活"')
    elif "NUM_" in button_name:
        number = button_name.split('_')[1]
        log_to_screen(f"用户按下了数字: {number}")
    elif button_name == "EXIT":
        log_to_screen("执行退出/重置程序...")
        send_command('t0.txt="识别图像"')
        # 清空日志屏幕
        log_buffer.clear()
        send_command('t_log.txt=""')


def serial_listener_thread():
    """后台线程：专门监听来自屏幕的按键消息"""
    log_to_screen("串口监听线程已启动...")
    while True:
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                for i in range(0, len(data), 7):
                    chunk = data[i:i + 7]
                    if len(chunk) == 7 and data[0] == 0x65 and data[3] == 0x01:
                        component_id = data[2]
                        if component_id in BUTTON_MAP:
                            handle_button_press(BUTTON_MAP[component_id])
            time.sleep(0.05)
        except Exception as e:
            log_to_screen(f"串口监听出错: {e}")
            break


# --- 线程二：数据更新发送 ---
def data_updater_thread():
    """后台线程：专门定时向屏幕发送更新数据"""
    log_to_screen("数据更新线程已启动...")
    max_power_val = 0.0
    while True:
        try:
            current_val = random.uniform(0.5, 1.5)
            power_val = 12.0 * current_val + random.uniform(-0.5, 0.5)
            if power_val > max_power_val: max_power_val = power_val

            send_command(f't_cur.txt="Current:{current_val:.3f}A"')
            send_command(f't_pow.txt="Power:{power_val:.3f}W"')
            send_command(f't_maxpower.txt="最大功耗:{max_power_val:.3f}W"')

            time.sleep(2)
        except Exception as e:
            log_to_screen(f"数据更新出错: {e}")
            break


# --- 线程安全的发送函数 ---
def send_command(cmd_str):
    """发送普通文本指令到屏幕的辅助函数 (线程安全)"""
    with serial_lock:
        time.sleep(0.02)
        ser.write(cmd_str.encode('gb2312'))
        ser.write(b'\xff\xff\xff')


# --- 主程序入口 ---
if __name__ == "__main__":
    listener = threading.Thread(target=serial_listener_thread, daemon=True)
    updater = threading.Thread(target=data_updater_thread, daemon=True)

    listener.start()
    updater.start()

    log_to_screen("主程序启动，所有线程已运行。")
    print("在终端窗口中按 Ctrl+C 退出程序。")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序退出。")
    finally:
        if ser.is_open:
            ser.close()
