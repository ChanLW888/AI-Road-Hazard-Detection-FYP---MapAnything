from ultralytics import YOLO

model = YOLO("../weights/best.pt")

print(model.task)
print(model.names)