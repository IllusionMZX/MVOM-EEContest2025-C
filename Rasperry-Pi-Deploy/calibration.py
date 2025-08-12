import cv2
import numpy as np

# 实际距离（厘米）
KNOWN_DISTANCE = 167.0
# A4纸边框实际宽度（厘米）
REAL_WIDTH = 2.0

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

def main():
    import threading
    import sys
    
    cap = cv2.VideoCapture(0)  # 修改为你的摄像头序号
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("请将A4纸放在167cm处，对准摄像头")
    print("在终端输入 'q' 并按回车键进行校准...")
    
    # 用于线程间通信的标志
    calibrate_flag = False
    current_pixel_width = 0
    
    def input_thread():
        nonlocal calibrate_flag, current_pixel_width
        while True:
            user_input = input().strip().lower()
            if user_input == 'q':
                calibrate_flag = True
                break
    
    # 启动输入线程
    input_thread_obj = threading.Thread(target=input_thread, daemon=True)
    input_thread_obj.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = find_largest_rectangle(contours)

        display_frame = frame.copy()
        pixel_width = 0

        if rect is not None:
            cv2.drawContours(display_frame, [rect], -1, (0, 255, 0), 3)
            pts = rect.reshape(4, 2)
            edge1 = np.linalg.norm(pts[0] - pts[1])
            edge2 = np.linalg.norm(pts[1] - pts[2])
            pixel_width = max(edge1, edge2)
            current_pixel_width = pixel_width
            cv2.putText(display_frame, f"Pixel width: {pixel_width:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # cv2.imshow("Calibration", display_frame)
        # cv2.imshow("Binary", binary)
        print(f"[CALIBRATION] Calibration frame: {display_frame.shape}")
        print(f"[CALIBRATION] Binary image: {binary.shape}")
        if pixel_width > 0:
            print(f"[CALIBRATION] Current pixel width: {pixel_width:.2f}")
        else:
            print("[CALIBRATION] No A4 paper detected - please adjust position")

        # 检查校准标志
        if calibrate_flag:
            # 计算focal_length
            if current_pixel_width > 0:
                focal_length = (current_pixel_width * KNOWN_DISTANCE) / REAL_WIDTH
                print(f"标定结果：pixel_width = {current_pixel_width:.2f}")
                print(f"计算得到FOCAL_LENGTH = {focal_length:.2f}")
                
                # 保存焦距到文件
                with open('focal_length.dat', 'w') as f:
                    f.write(str(focal_length))
                print(f"焦距已保存到 focal_length.dat 文件")
            else:
                print("未检测到有效边框，请调整A4纸或摄像头位置后重试。")
            break

        # 短暂延时
        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()