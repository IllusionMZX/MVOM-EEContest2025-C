import onnxruntime as ort
import numpy as np
import cv2
import math
import os
from serial_utils import send_command, log_to_screen

class OverlappingSquareDetector:
    def __init__(self, model_path):
        """
        重叠正方形检测器，使用ONNX模型进行实例分割。

        Args:
            model_path: 训练好的ONNX模型路径
        """
        # A4 paper dimensions in cm, used for pixel-to-cm conversion
        self.a4_width_cm = 21.0
        self.a4_height_cm = 29.7
        
        self.session = self._load_onnx_model(model_path)
        print(f"[INFO] ONNX model '{model_path}' loaded successfully.")

    def _load_onnx_model(self, model_path):
        """
        Loads the ONNX model using ONNX Runtime.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            return session
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

    def _preprocess_image(self, image, input_shape=(640, 640)):
        """
        Prepares the image for model inference.
        """
        original_h, original_w = image.shape[:2]
        resized_img = cv2.resize(image, input_shape, interpolation=cv2.INTER_AREA)
        
        input_data = resized_img.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        return input_data, (original_w, original_h)

    def _postprocess_yolov8_seg(self, outputs, original_size, input_shape=(640, 640), conf_threshold=0.5, iou_threshold=0.45):
        """
        Processes the raw model outputs to get masks and bounding boxes.
        """
        predictions = outputs[0]
        proto = outputs[1]
        
        boxes_raw = predictions[0, :4, :]
        confidences_raw = predictions[0, 4, :]
        mask_coeffs_raw = predictions[0, 5:, :]
        
        boxes = boxes_raw.T
        confidences = confidences_raw
        mask_coeffs = mask_coeffs_raw.T
        
        valid_indices = np.where(confidences > conf_threshold)[0]
        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]
        mask_coeffs = mask_coeffs[valid_indices]
        
        if len(valid_indices) == 0:
            return [], []
        
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=np.vstack((x1, y1, x2, y2)).T.astype(np.float32).tolist(), 
            scores=confidences.tolist(), 
            score_threshold=conf_threshold, 
            nms_threshold=iou_threshold
        )
        
        if len(nms_indices) == 0:
            return [], []
        
        nms_indices = nms_indices.flatten()
        
        nms_indices = nms_indices[nms_indices < len(boxes)]
        boxes = boxes[nms_indices]
        confidences = confidences[nms_indices]
        mask_coeffs = mask_coeffs[nms_indices]

        masks = np.matmul(mask_coeffs, proto[0].reshape(proto.shape[1], -1))
        masks = masks.reshape(-1, proto.shape[2], proto.shape[3])
        
        final_masks = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = cv2.resize(mask, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
            final_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_AREA)
            final_masks.append(final_mask)
            
        return boxes, final_masks

    def detect_overlapping_squares(self, image, focal_length, distance):
        """
        检测重叠正方形的主函数，使用ONNX模型。
        
        Args:
            image: 输入图像
            focal_length: 相机焦距 (此处未使用)
            distance: 相机到A4纸的距离 (此处未使用)
            
        Returns:
            result_image: 标注了检测结果的图像
            squares_info: 检测到的正方形信息列表
        """
        print(f"\n[INFO] ========== Starting ONNX Inference for Overlapping Squares ==========")
        print(f"[INFO] 距离：{distance}cm")
        log_to_screen(f"[INFO] 距离：{distance}cm")

        original_h, original_w = image.shape[:2]
        
        # Calculate pixel to cm ratio based on A4 paper dimensions
        px_per_cm_width = original_w / self.a4_width_cm
        px_per_cm_height = original_h / self.a4_height_cm
        px_per_cm = (px_per_cm_width + px_per_cm_height) / 2

        # --- 1. Run Inference ---
        input_data, original_size = self._preprocess_image(image)
        outputs = self.session.run(None, {'images': input_data})
        boxes, masks = self._postprocess_yolov8_seg(outputs, original_size)

        if not masks:
            print("[INFO] No squares detected by the model.")
            log_to_screen("[INFO] No squares detected by the model.")
            return image.copy(), []

        # --- 2. Analyze and Draw Results ---
        result_image = image.copy()
        squares_info = []
        min_area_square_index = -1
        min_area = float('inf')

        for i, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            rotated_rect = cv2.minAreaRect(largest_contour)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = box_points.astype(np.int32)
            
            width_px, height_px = rotated_rect[1]
            avg_edge_pixel = (width_px + height_px) / 2
            
            edge_length_cm = avg_edge_pixel / px_per_cm
            area_cm2 = edge_length_cm ** 2

            if area_cm2 < min_area:
                min_area = area_cm2
                min_area_square_index = i

            squares_info.append({
                'id': i + 1,
                'corners': box_points.tolist(),
                'edge_length_cm': edge_length_cm,
                'area_cm2': area_cm2,
                'is_smallest': False
            })

        if min_area_square_index != -1:
            squares_info[min_area_square_index]['is_smallest'] = True
            smallest_info = squares_info[min_area_square_index]
            print(f"[INFO] 最小边长：{smallest_info['edge_length_cm']:.2f}cm")
            print(f"[INFO] 最小面积：{smallest_info['area_cm2']:.2f}cm^2")
            log_to_screen(f"[INFO] 最小边长：{smallest_info['edge_length_cm']:.2f}cm")
            log_to_screen(f"[INFO] 最小面积：{smallest_info['area_cm2']:.2f}cm^2")
            log_to_screen(f"[INFO] 最小正方形信息：{smallest_info}")
            log_to_screen(f"[INFO] 距离：{distance:.2f}cm")
            send_command(f't_x.txt="最小边长:{smallest_info["edge_length_cm"]:.2f}cm"')
            send_command(f't_shape.txt="形状:正方形"')
            send_command(f't_distance.txt="距离:{distance:.2f}cm"')

        return result_image, squares_info

# The main interface function as required by the user
def detect_overlapping_squares(image, focal_length, distance):
    """
    Main function to detect overlapping squares using a pre-trained ONNX model.
    """
    model_path = "best.onnx"
    detector = OverlappingSquareDetector(model_path)
    return detector.detect_overlapping_squares(image, focal_length, distance)