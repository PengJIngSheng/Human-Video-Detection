from ultralytics import YOLO


def train():
    print("正在加载 YOLOv13 模型...")
    model = YOLO('models/yolov13x.pt')

    print("开始训练...")
    results = model.train(
        data='data/staff_dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=2,
        device='0',
        project='staff_project',
        name='yolov13_finetune',
        exist_ok=True,
        plots=True
    )

    print("训练完成！")
    print(f"最佳模型路径: staff_project/yolov13_finetune/weights/best.pt")


if __name__ == '__main__':
    train()