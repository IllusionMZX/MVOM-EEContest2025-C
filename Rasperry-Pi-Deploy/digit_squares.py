import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from serial_utils import send_command, log_to_screen

# --- 加载 ONNX 模型 ---
onnx_model_path = "lenet5_mnist_model.onnx"

try:
    session_options = ort.SessionOptions()
    session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
except Exception as e:
    session = None

input_name = session.get_inputs()[0].name if session else None
output_name = session.get_outputs()[0].name if session else None

# --- 准备图像预处理函数 ---
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def infer_digit_from_roi(roi_image):
    """使用ONNX模型识别ROI中的数字"""
    if session is None:
        return -1, 0.0
    
    try:
        if len(roi_image.shape) == 3:
            roi_pil = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)).convert('L')
        else:
            roi_pil = Image.fromarray(roi_image).convert('L')
        
        roi_pil = roi_pil.resize((28, 28), Image.LANCZOS)
        
        img_array = np.array(roi_pil)
        avg_brightness = np.mean(img_array)
        
        if avg_brightness > 127.5:
            roi_pil = Image.fromarray(255 - img_array)
        
        input_tensor = preprocess(roi_pil)
        input_tensor = input_tensor.unsqueeze(0)
        
        ort_inputs = {input_name: input_tensor.numpy()}
        ort_outputs = session.run([output_name], ort_inputs)
        
        output_logits = ort_outputs[0]
        output_probs = np.exp(output_logits) / np.sum(np.exp(output_logits), axis=1, keepdims=True)
        
        predicted_digit = np.argmax(output_logits)
        confidence = np.max(output_probs)
        
        return int(predicted_digit), float(confidence)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return -1, 0.0

BORDER_WIDTH_CM = 2.0

def detect_digit_squares(roi, target_digit, focal_length, real_distance):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = roi.shape[:2]

    border_ratio = 2.0 / 21.0
    border_pixel_width = w * border_ratio
    pixel_to_cm_ratio = BORDER_WIDTH_CM / border_pixel_width

    target_squares = []
    valid_contours = []
    all_candidates = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        margin = 10
        is_border_contour = (x <= margin or y <= margin or 
                           (x + w_cnt) >= (w - margin) or 
                           (y + h_cnt) >= (h - margin))
        if is_border_contour:
            continue
        min_area = 200
        max_area = w * h * 0.15
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter * perimeter)
                if compactness > 0.1:
                    valid_contours.append((cnt, area, i))

    for cnt, area, contour_idx in valid_contours:
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect) if min(w_rect, h_rect) > 0 else float('inf')
        
        if aspect_ratio < 2.0:
            margin = 3
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + w_rect + margin)
            y2 = min(h, y + h_rect + margin)
            square_roi = gray[y1:y2, x1:x2]
            
            digit = -1
            confidence = 0.0
            try:
                square_binary = cv2.adaptiveThreshold(square_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                     cv2.THRESH_BINARY, 11, 2)
                
                kernel = np.ones((2,2), np.uint8)
                square_binary = cv2.morphologyEx(square_binary, cv2.MORPH_CLOSE, kernel)
                
                digit, confidence = infer_digit_from_roi(square_binary)
                
            except Exception as e:
                pass
            
            if digit != -1:
                perimeter = cv2.arcLength(cnt, True)
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 4:
                    if len(approx) != 4:
                        epsilon = 0.05 * perimeter
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) != 4:
                        approx = np.array([[x, y], [x + w_rect, y], 
                                         [x + w_rect, y + h_rect], [x, y + h_rect]], dtype=np.int32)
                        approx = approx.reshape(-1, 1, 2)
                    pts = approx.reshape(4, 2)
                    edges = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
                    avg_edge_pixels = np.mean(edges)
                    edge_length_cm = avg_edge_pixels * pixel_to_cm_ratio
                    
                    if 5.0 <= edge_length_cm <= 15.0:
                        candidate = {
                            'contour': approx,
                            'digit': digit,
                            'confidence': confidence,
                            'edge_length_cm': edge_length_cm,
                            'edge_length_pixels': avg_edge_pixels,
                            'edges': edges,
                            'center': (x + w_rect//2, y + h_rect//2),
                            'area': area,
                            'matches_target': (digit == target_digit)
                        }
                        
                        all_candidates.append(candidate)

    target_matches = [c for c in all_candidates if c['matches_target'] and c['confidence'] > 0.2]
    
    if target_matches:
        best_candidate = max(target_matches, key=lambda x: x['confidence'])
    else:
        if all_candidates:
            best_candidate = max(all_candidates, key=lambda x: x['confidence'])
        else:
            best_candidate = None
    
    if best_candidate:
        target_squares = [best_candidate]
    else:
        target_squares = []

    # Print and log information for each identified square
    for i, square in enumerate(target_squares):
        area_cm2 = square['area'] * (pixel_to_cm_ratio ** 2)
        message = (
            f"正方形{i+1}: "
            f"识别数字: {square['digit']}, "
            f"置信度: {square['confidence']:.2f}, "
            f"边长: {square['edge_length_cm']:.1f}cm, "
            f"面积: {area_cm2:.1f}cm^2"
        )
        print(message)
        log_to_screen(f"边长: {square['edge_length_cm']:.1f}cm")
        log_to_screen(f"面积: {area_cm2:.1f}cm^2")
        send_command(f't_x.txt="边长:{square["edge_length_cm"]:.2f}cm"')
        send_command(f't_shape.txt="形状:正方形"')
        send_command(f't_distance.txt="距离:{real_distance:.2f}cm"')


    return target_squares