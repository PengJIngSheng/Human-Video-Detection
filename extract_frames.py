import cv2
import os

VIDEO_PATH = 'data/input/sample.mp4'
TRAIN_DIR = 'data/dataset/images/train'
VAL_DIR = 'data/dataset/images/val'


def extract_frames():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {VIDEO_PATH}")
        return

    frame_count = 0
    saved_count = 0

    print(f"开始从 {VIDEO_PATH} 提取图片...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:

            filename = f"frame_{saved_count:04d}.jpg"

            if saved_count % 5 == 0:
                save_path = os.path.join(VAL_DIR, filename)
            else:
                save_path = os.path.join(TRAIN_DIR, filename)

            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print("-" * 30)
    print(f"提取完成！")
    print(f"共处理帧数: {frame_count}")
    print(f"共保存图片: {saved_count} 张")
    print(f"查看目录: data/dataset/images/")


if __name__ == "__main__":
    extract_frames()