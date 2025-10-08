from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    segment_poly = Polygon(segment)
    min_dist = float('inf')
    for bbox in bboxes:
        gun_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
        dist = segment_poly.distance(gun_box)
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            matched_box = bbox

    return matched_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_img = image_array.copy()
    overlay = annotated_img.copy()
    alpha = 0.4  # Ajusta la transparencia de la imagen con el color de segmentación
    for poly, label, bbox in zip(segmentation.polygons, segmentation.labels, segmentation.boxes):
        color = (0, 255, 0) if label == 'safe' else (0, 0, 255)
        pts = np.array(poly, np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
        if draw_boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )       
    annotated_img = cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0)
    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        seg_results = self.seg_model(image_array, conf=threshold, task='segment')[0]
        polygons = []
        boxes = []
        labels = []
        # Detecta armas
        detection = self.detect_guns(image_array, threshold)
        gun_boxes = detection.boxes
        for mask, box, cls in zip(seg_results.masks.xy, seg_results.boxes.xyxy.tolist(), seg_results.boxes.cls.tolist()):
            if cls == 0:
                poly = [[int(x), int(y)] for x, y in mask]
                polygons.append(poly)
                bbox = [int(v) for v in box]
                boxes.append(bbox)
                # Empareja con un arma
                gun_match = match_gun_bbox(poly, gun_boxes, max_distance)
                label = 'danger' if gun_match else 'safe'
                labels.append(label)

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(polygons),
            polygons=polygons,
            boxes=boxes,
            labels=labels
        )
