from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-obb.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-obb.yaml").load(
#     "yolo11n.pt"
# )  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="dota8.yaml", epochs=100, imgsz=640)

# Validate the model
# metrics = model.val(
#     data="dota8.yaml"
# )  # no arguments needed, dataset and settings remembered
# print(metrics.box.map)  # map50-95(B)
# print(metrics.box.map50)  # map50(B)
# print(metrics.box.map75)  # map75(B)
# print(metrics.box.maps)  # a list containing mAP50-95(B) for each category

model = YOLO(
    "models/yolo11n-obb.pt"
)  # load a pretrained model (recommended for training)


# Predict with the model
results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image

# Access the results
for result in results:
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [
        result.names[cls.item()] for cls in result.obb.cls.int()
    ]  # class name of each box
    confs = result.obb.conf  # confidence score of each box
    result.show()  # display results

# Export the model
model.export(format="onnx")
