"""
YOLO26-L 학습

실행:
  python train_yolo26l.py
"""

from ultralytics import YOLO

# ── 설정 ──
YOLO_DIR = r"C:\github\VLA\yolo_format"
DATA_YAML = f"{YOLO_DIR}/data.yaml"
EPOCHS = 100
BATCH = 16
IMGSZ = 640
DEVICE = 0
PROJECT = "./runs_compare"
NAME = "yolo26l"

# ── 학습 ──
model = YOLO("yolo26l.pt")

model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    device=DEVICE,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    # 소형 객체 최적화
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    # 기본 augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    # 학습 안정화
    warmup_epochs=3,
    patience=20,
    save_period=10,
)

# ── 평가 ──
metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, device=DEVICE)
print(f"\n=== YOLO26-L 결과 ===")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")