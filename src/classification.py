from ultralytics import YOLO

# Load a model

model = YOLO(
    "models/yolo11n-cls.pt"
)  # load a pretrained model (recommended for training)

# Validate the model
metrics = model.val(
    data="mnist160"
)  # no arguments needed, dataset and settings remembered
print(metrics.top1)  # top1 accuracy
print(metrics.top5)  # top5 accuracy

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

for result in results:
    result.show()  # display results

# Export the model
model.export(format="onnx")
