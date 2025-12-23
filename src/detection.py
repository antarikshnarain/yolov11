from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml").load(
    "models/yolo11n.pt"
)  # build from YAML and transfer weights

# Train the model
results = model.train(data="models/coco.yaml", epochs=100, imgsz=640)

# Load a model
model = YOLO("models/yolo11n.pt")  # load an official model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list containing mAP50-95 for each category


# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [
        result.names[cls.item()] for cls in result.boxes.cls.int()
    ]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box

# Export the model
model.export(format="onnx")
