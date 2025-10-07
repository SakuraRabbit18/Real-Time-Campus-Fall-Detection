from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov12-fall-detection.yaml")
# Or load the model from the .pt file
model = YOLO("yolov12-fall-detection.pt")

# Train the model
# Change the data.yaml path to your own data.yaml path
results = model.train(data="data.yaml", epochs=200)
