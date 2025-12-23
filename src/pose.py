from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n-pose.pt")  # load an official model

# Validate the model (ensure the YAML path is correct and exists)
metrics = model.val(data="coco8-pose.yaml")  # specify 'data' argument for clarity

print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list containing mAP50-95 for each category
print(metrics.pose.map)  # map50-95(P)
print(metrics.pose.map50)  # map50(P)
print(metrics.pose.map75)  # map75(P)
print(metrics.pose.maps)  # a list containing mAP50-95(P) for each category

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    result.show()  # display results

# Export the model
model.export(format="onnx")
