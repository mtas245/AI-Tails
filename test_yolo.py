from ultralytics import YOLO

model = YOLO("models/cat_face_yolov8n.pt")

results = model.predict(
    source="test/test_cat.jpg",
    conf=0.25,
    save=True
)

for result in results:
    print(result.boxes)