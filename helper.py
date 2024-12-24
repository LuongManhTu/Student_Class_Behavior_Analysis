import cv2
import numpy as np
import groundingdino.datasets.transforms as T
import torch
from typing import Tuple
from PIL import Image

def load_image_from_frame(image_frame: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Args:
        image_frame (np.ndarray): Frame ảnh dưới dạng numpy array (H, W, C).

    Returns:
        image (np.ndarray): Ảnh gốc dưới dạng numpy array.
        image_transformed (torch.Tensor): Ảnh đã qua transform, sẵn sàng cho mô hình.
    """
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_source = Image.fromarray(image_frame.astype('uint8'), 'RGB')
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)

    return image, image_transformed

def denormalize_boxes(boxes, image_width, image_height):
    boxes_absolute = []
    for cx, cy, w, h in boxes:
        xmin = (cx - w / 2) * image_width
        xmax = (cx + w / 2) * image_width
        ymin = (cy - h / 2) * image_height
        ymax = (cy + h / 2) * image_height
        boxes_absolute.append([xmin, ymin, xmax, ymax])
    return boxes_absolute

def post_process(boxes, scores, size_factor=1.5, nms_threshold=0.3):
    """
    Lọc các bounding boxes có kích thước quá lớn hoặc nhỏ so với trung bình.
    Thực hiện Non-Maximum Suppression (NMS) để lọc các bounding boxes mà không cần scores.
    
    Args:
        boxes (list of list): Danh sách các bounding boxes, mỗi bounding box được biểu diễn
                               dưới dạng [x1, y1, x2, y2].
        threshold (float): Ngưỡng IoU để loại bỏ các bounding boxes chồng lấp.

    Returns:
        list: Các bounding boxes đã được lọc.
    """
    if len(boxes) == 0:
        return []

    ####      REMOVE ANOMALIES
    # Chuyển đổi dữ liệu thành mảng numpy
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Tính toán diện tích của các bounding boxes
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1
    areas = widths * heights

    # Tính toán diện tích trung bình và độ lệch chuẩn
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    # Xác định ngưỡng cho kích thước lớn và nhỏ
    min_area = mean_area - size_factor * std_area
    max_area = mean_area + size_factor * std_area
    
    # Lọc các bounding boxes
    filtered_boxes = boxes[(areas >= min_area) & (areas <= max_area)]

    
    ####      NMS
    # Tính toán diện tích của các bounding boxes
    x1 = filtered_boxes[:, 0]
    y1 = filtered_boxes[:, 1]
    x2 = filtered_boxes[:, 2]
    y2 = filtered_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        # Lấy bounding box có score cao nhất
        i = order[0]
        keep.append(i)

        # Tính toán IoU với bounding boxes còn lại
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter_area = w * h

        # Tính toán IoU
        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)

        # Giữ lại những bounding boxes có IoU nhỏ hơn ngưỡng
        indices = np.where(iou <= nms_threshold)[0] + 1
        order = order[indices]

    return torch.tensor(keep, dtype=torch.float32)






