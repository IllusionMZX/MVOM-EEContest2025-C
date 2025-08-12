import cv2
import numpy as np

# A4纸的实际宽度和高度（厘米）
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

from serial_utils import send_command, log_to_screen

def detect_multiple_squares(roi, focal_length, real_distance):
    """
    检测图像中的多个正方形，并打印所有正方形的边长和面积。
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = roi.shape[:2]
    
    # --- 修正像素到厘米的比例计算 ---
    # 假设roi是已经校正过的A4纸图像
    pixel_to_cm_ratio = A4_WIDTH_CM / w
    
    squares = []
    result_roi = roi.copy()
    valid_contours = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        
        # 修正边界过滤逻辑，避免将小物体误滤除
        margin = 5
        is_border_contour = (x <= margin or y <= margin or (x + w_cnt) >= (w - margin) or (y + h_cnt) >= (h - margin))
        if is_border_contour:
            # 只过滤掉过大的边框轮廓
            if area > w * h * 0.5:
                continue
        
        # 降低最小面积要求，避免小正方形被过滤
        min_area = 100 
        max_area = w * h * 0.5 # 允许更大的正方形
        
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                # 正方形的紧凑度应该接近于0.785
                compactness = (4 * np.pi * area) / (perimeter * perimeter)
                if compactness > 0.6:  # 提高紧凑度阈值，更倾向于正方形
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.85: # 确保轮廓是实心的
                            valid_contours.append((cnt, area, i))

    for cnt, area, contour_idx in valid_contours:
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect) if min(w_rect, h_rect) > 0 else float('inf')
        
        if aspect_ratio < 2.0: # 过滤掉太长的矩形
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            if len(approx) == 4: # 确保是四边形
                pts = approx.reshape(4, 2)
                edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
                avg_edge_pixels = np.mean(edges)
                edge_length_cm = avg_edge_pixels * pixel_to_cm_ratio
                
                edge_ratio = max(edges) / min(edges) if min(edges) > 0 else float('inf')
                
                if edge_ratio < 1.3: # 确保四边形边长接近
                    # 降低最小边长要求
                    if 0.5 <= edge_length_cm <= 15.0: 
                        angles = []
                        for i in range(4):
                            p1 = pts[i]
                            p2 = pts[(i+1)%4]
                            p3 = pts[(i+2)%4]
                            v1 = p1 - p2
                            v2 = p3 - p2
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.degrees(np.arccos(cos_angle))
                            angles.append(angle)
                        angle_deviations = [abs(angle - 90) for angle in angles]
                        max_deviation = max(angle_deviations)
                        
                        if max_deviation < 20: # 确保是直角
                            squares.append({
                                'contour': approx,
                                'edge_length_cm': edge_length_cm,
                                'area_cm2': area * (pixel_to_cm_ratio ** 2)
                            })
                            cv2.drawContours(result_roi, [approx], -1, (0, 255, 0), 2)
                            center = (int(x + w_rect/2), int(y + h_rect/2))
                            cv2.putText(result_roi, f"{edge_length_cm:.1f}cm", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 打印所有检测到的正方形
    log_to_screen(f"找到 {len(squares)} 个正方形。")
    if squares:
        for i, square in enumerate(squares):
            log_to_screen(f"正方形{i+1}: 边长: {square['edge_length_cm']:.1f}cm, 面积: {square['area_cm2']:.1f}cm^2")
    
    # 找到最小面积的正方形
    min_area_square = None
    if squares:
        min_area_square = min(squares, key=lambda s: s['area_cm2'])
    
    # 打印最小正方形的信息
    log_to_screen(f"距离: {real_distance:.2f}cm")
    send_command(f't_distance.txt="距离:{real_distance:.2f}cm"')
    send_command('t_shape.txt="形状:正方形"')
    if min_area_square:
        log_to_screen(f"最小正方形边长: {min_area_square['edge_length_cm']:.1f}cm")
        log_to_screen(f"最小正方形面积: {min_area_square['area_cm2']:.1f}cm^2")
        send_command(f't_x.txt="最小边长:{min_area_square["edge_length_cm"]:.2f}cm"')
    else:
        send_command('t_x.txt="边长/直径:未找到"')
        
    return result_roi, squares