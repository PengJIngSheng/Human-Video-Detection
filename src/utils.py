import json
import os
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def create_directories(paths: List[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_video(video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

    return cap, metadata


def save_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def draw_detections(frame: np.ndarray,
                    detections: List[Dict],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2,
                    font_scale: float = 0.6,
                    draw_confidence: bool = True,
                    draw_id: bool = True) -> np.ndarray:
    annotated = frame.copy()

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        labels = []
        if draw_id and 'staff_id' in det:
            labels.append(f"ID: {det['staff_id']}")
        if draw_confidence and 'confidence' in det:
            labels.append(f"{det['confidence']:.2f}")

        label = " | ".join(labels)

        if label:
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            cv2.rectangle(annotated,
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)

            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 0, 0), thickness)

    return annotated


def bbox_center(bbox) -> Tuple[float, float]:
    if isinstance(bbox, (list, tuple)):
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
    elif hasattr(bbox, 'tolist'):
        bbox_list = bbox.tolist()
        x1, y1, x2, y2 = bbox_list
    else:
        raise TypeError(f"bbox must be list, tuple, or numpy array, got {type(bbox)}")

    return ((x1 + x2) / 2, (y1 + y2) / 2)


def save_detection_results(results: Dict, output_path: str) -> None:
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def format_detection_result(frame_idx: int,
                            detections: List[Dict],
                            has_staff: bool = True) -> Dict:
    coordinates = []

    for det in detections:
        bbox = det['bbox']
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        elif not isinstance(bbox, list):
            bbox = list(bbox)

        center = bbox_center(bbox)

        coord = {
            'staff_id': det.get('staff_id', -1),
            'bbox': [float(x) for x in bbox],  # [x1, y1, x2, y2]
            'center': [float(center[0]), float(center[1])],  # [x, y]
            'confidence': float(det.get('confidence', 0.0))
        }
        coordinates.append(coord)

    return {
        'frame': frame_idx,
        'staff_present': has_staff,
        'staff_count': len(detections),
        'coordinates': coordinates
    }


def create_summary_report(detections: List[Dict],
                         video_name: str,
                         total_frames: int) -> Dict:
    frames_with_staff = sum(1 for d in detections if d['staff_present'])
    total_staff_detections = sum(d['staff_count'] for d in detections)

    all_confidences = []
    for d in detections:
        for coord in d['coordinates']:
            all_confidences.append(coord['confidence'])

    avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

    return {
        'video': video_name,
        'total_frames': total_frames,
        'frames_with_staff': frames_with_staff,
        'detection_rate': frames_with_staff / total_frames if total_frames > 0 else 0,
        'total_detections': total_staff_detections,
        'average_confidence': float(avg_confidence),
        'detections': detections
    }


def print_progress(current: int, total: int, prefix: str = 'Progress') -> None:
    percentage = 100 * (current / float(total))
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print(f'\r{prefix}: |{bar}| {percentage:.1f}% ({current}/{total})', end='')

    if current == total:
        print()