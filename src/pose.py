from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
# model = YOLO(
#     "models/yolo11n-pose.pt"
# )  # load a pretrained model (recommended for training)
# # model = YOLO("yolo11n-pose.yaml").load(
# #     "models/yolo11n-pose.pt"
# # )  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)

model = YOLO("models/yolo11n-pose.pt")  # load an official model
# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# print(metrics.box.map)  # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75
# print(metrics.box.maps)  # a list containing mAP50-95 for each category
# print(metrics.pose.map)  # map50-95(P)
# print(metrics.pose.map50)  # map50(P)
# print(metrics.pose.map75)  # map75(P)
# print(metrics.pose.maps)  # a list containing mAP50-95(P) for each category

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
