from ultralytics import YOLO

# This will download the yolov8n.pt weights if not already present
YOLO("yolov8n.pt")
print("Downloaded yolov8n.pt")
