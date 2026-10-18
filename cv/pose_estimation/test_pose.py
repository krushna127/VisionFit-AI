from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

print("yolo-pose model load successfully!")
print("Model task: ", model.task)