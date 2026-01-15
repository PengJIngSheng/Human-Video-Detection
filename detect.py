import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from src.detector import StaffDetector
from src.tracker import StaffTracker
from src.utils import (
    create_directories,
    load_video,
    save_video_writer,
    draw_detections,
    save_detection_results,
    format_detection_result,
    create_summary_report
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Staff Detection and Tracking')
    parser.add_argument('--video', type=str, default='data/input/sample.mp4',
                        help='Path to input video')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--conf', type=float, default=None,
                        help='Confidence threshold (overrides config)')
    parser.add_argument('--iou', type=float, default=None,
                        help='IOU threshold (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu (overrides config)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save annotated video')
    parser.add_argument('--no-track', action='store_true',
                        help='Disable tracking (detection only)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show real-time visualization')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    config = load_config(args.config)

    if args.weights:
        config['model']['weights'] = args.weights
    if args.output:
        config['paths']['output'] = args.output
    if args.conf:
        config['detection']['conf_threshold'] = args.conf
    if args.iou:
        config['detection']['iou_threshold'] = args.iou
    if args.device:
        config['model']['device'] = args.device
    if args.save_video:
        config['output']['save_video'] = True
    if args.no_track:
        config['tracking']['enabled'] = False

    output_dir = Path(config['paths']['output'])
    create_directories([str(output_dir)])

    print("Initializing Staff Detector...")
    detector = StaffDetector(
        weights_path=config['model']['weights'],
        device=config['model']['device'],
        conf_threshold=config['detection']['conf_threshold'],
        iou_threshold=config['detection']['iou_threshold'],
        half=config['model'].get('half', True)
    )

    tracker = None
    if config['tracking']['enabled']:
        print("\nInitializing Tracker...")
        tracker = StaffTracker(
            max_age=config['tracking']['track_buffer'],
            min_hits=1,
            iou_threshold=config['tracking']['match_thresh']
        )

    cap, metadata = load_video(args.video)

    print(f"Video Info:")

    video_writer = None
    if config['output']['save_video']:
        output_video_path = output_dir / 'result_video.mp4'
        video_writer = save_video_writer(
            str(output_video_path),
            metadata['fps'],
            metadata['width'],
            metadata['height']
        )
        print(f"\nOutput video will be saved to: {output_video_path}")

    print("\n" + "=" * 60)
    print("Processing Video...")

    frame_idx = 0
    all_detections = []

    pbar = tqdm(total=metadata['frame_count'], desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        detections = detector.detect_staff(frame)

        if tracker:
            if len(detections) > 0:
                det_array = np.array([
                    det['bbox'] + [det['confidence']]
                    for det in detections
                ])
            else:
                det_array = np.empty((0, 5))

            tracked = tracker.update(det_array)

        elif len(detections) > 0:
            tracked = detections
        else:
            tracked = []

        result = format_detection_result(
            frame_idx,
            tracked,
            has_staff=len(tracked) > 0
        )
        all_detections.append(result)

        if config['output']['save_video'] or args.visualize:
            annotated_frame = draw_detections(
                frame,
                tracked,
                color=tuple(config['output']['bbox_color']),
                thickness=config['output']['bbox_thickness'],
                font_scale=config['output']['font_scale'],
                draw_confidence=config['output']['draw_confidence'],
                draw_id=config['tracking']['enabled']
            )

            cv2.putText(
                annotated_frame,
                f"Frame: {frame_idx}/{metadata['frame_count']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Write to video
            if video_writer:
                video_writer.write(annotated_frame)

            # Display
            if args.visualize:
                cv2.imshow('Staff Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

        pbar.update(1)

    pbar.close()

    cap.release()
    if video_writer:
        video_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

    print("Saving Results...")

    summary = create_summary_report(
        all_detections,
        Path(args.video).name,
        metadata['frame_count']
    )

    output_json_path = output_dir / 'detection_results.json'
    save_detection_results(summary, str(output_json_path))

    print("\nProcessing complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
