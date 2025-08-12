import cv2
import numpy as np
from serial_utils import send_command, log_to_screen

def detect_shape_and_size(roi, focal_length, real_distance):
    """
    检测图像中的单个几何图形（圆形、三角形等）并计算其尺寸和距离。
    对于圆形，使用cv2.fitEllipse进行更精确的测量。
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    image_height = roi.shape[0]
    image_width = roi.shape[1]

    # --- 使用A4纸宽度（21.0cm）和图像像素宽度来计算像素-厘米比 ---
    # 这比基于边框的计算更直接且稳定
    A4_WIDTH_CM = 21.0
    pixel_to_cm_ratio = A4_WIDTH_CM / image_width

    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = roi.shape[:2]
    valid_contours = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        is_border_contour = (x <= 5 or y <= 5 or (x + w_cnt) >= (w - 5) or (y + h_cnt) >= (h - 5))
        if is_border_contour:
            continue
        min_area = 30
        max_area = w * h * 0.7
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter * perimeter)
                if compactness > 0.02:
                    valid_contours.append((cnt, area))
    if not valid_contours:
        log_to_screen("未找到有效形状。")
        send_command('t_shape.txt="形状:未找到"')
        send_command('t_x.txt="边长/直径:未找到"')
        send_command(f't_distance.txt="距离:{real_distance:.2f}cm"')
        return roi, None, None, None, None
    target_cnt, best_area = max(valid_contours, key=lambda x: x[1])
    epsilon = 0.02 * cv2.arcLength(target_cnt, True)
    approx = cv2.approxPolyDP(target_cnt, epsilon, True)
    shape_type = "未知"
    size_pixel = 0
    size_cm = 0
    if len(approx) == 3:
        shape_type = "三角形"
        pts = approx.reshape(3, 2)
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        size_pixel = max(a, b, c)
    elif len(approx) == 4:
        pts = approx.reshape(4, 2)
        edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        edge_ratio = max(edges) / min(edges)
        if edge_ratio < 1.3:
            shape_type = "正方形"
        else:
            shape_type = "矩形"
        size_pixel = max(edges)
    else:
        # 如果不是多边形，可能是圆形
        # --- 针对圆形进行修改：使用cv2.fitEllipse拟合椭圆 ---
        # 这种方法对噪声更鲁棒，能更精确地计算直径
        if len(target_cnt) >= 5: # 拟合椭圆需要至少5个点
            try:
                ellipse = cv2.fitEllipse(target_cnt)
                (x, y), (major_axis, minor_axis), angle = ellipse
                # 取长轴和短轴的平均值作为直径，以获得更稳定的结果
                pixel_diameter = (major_axis + minor_axis) / 2
                pixel_width = pixel_diameter
                # 检查形状是否更像圆形
                if abs(major_axis - minor_axis) / major_axis < 0.2: # 形状接近圆形
                    shape_type = "圆形"
                    size_pixel = pixel_diameter
                else:
                    shape_type = "不规则"
                    size_pixel = max(major_axis, minor_axis) # 使用最长边作为尺寸
            except cv2.error:
                # 如果拟合失败，则退回使用最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(target_cnt)
                shape_type = "圆形"
                size_pixel = radius * 2
                log_to_screen("椭圆拟合失败，退回到最小外接圆。")
        else:
            (x, y), radius = cv2.minEnclosingCircle(target_cnt)
            shape_type = "圆形"
            size_pixel = radius * 2
    
    size_cm = size_pixel * pixel_to_cm_ratio
    
    log_to_screen(f"形状: {shape_type}")
    log_to_screen(f"大小: {size_cm:.2f}cm")
    log_to_screen(f"距离: {real_distance:.2f}cm")
    send_command(f't_shape.txt="形状:{shape_type}"')
    send_command(f't_x.txt="边长/直径:{size_cm:.2f}cm"')
    send_command(f't_distance.txt="距离:{real_distance:.2f}cm"')
    
    return roi, shape_type, size_cm, size_pixel, best_area