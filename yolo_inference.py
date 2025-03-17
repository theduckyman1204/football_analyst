from ultralytics import YOLO

# Load mô hình đã train
model = YOLO('models/bestv5.pt')

# Thực hiện đánh giá trên tập validation
metrics = model.val()

# In ra các chỉ số chính
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.precision:.4f}")
print(f"Recall: {metrics.box.recall:.4f}")
print(f"F1 Score: {metrics.box.f1:.4f}")
print(f"Number of detected objects: {metrics.box.n:.0f}")