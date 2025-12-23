from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.yaml").load(
#     "yolo11n.pt"
# )  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)

# Load a model
model = YOLO("models/yolo11n-seg.pt")  # load an official model

# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# print(metrics.box.map)  # map50-95(B)
# print(metrics.box.map50)  # map50(B)
# print(metrics.box.map75)  # map75(B)
# print(metrics.box.maps)  # a list containing mAP50-95(B) for each category
# print(metrics.seg.map)  # map50-95(M)
# print(metrics.seg.map50)  # map50(M)
# print(metrics.seg.map75)  # map75(M)
# print(metrics.seg.maps)  # a list containing mAP50-95(M) for each category

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    result.show()  # display results

# Export the model
model.export(format="onnx")
