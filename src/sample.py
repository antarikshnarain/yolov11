from ultralytics import YOLO

# 1. Load a pretrained YOLO11 model (n, s, m, l, or x)
model = YOLO("models/yolo11n.pt")

# 2. Run inference on an image (can be a local path, URL, or even a video)
results = model("https://ultralytics.com/images/bus.jpg")

# 3. Process and show results
for result in results:
    result.show()  # Opens a window showing detected boxes and labels

    # Optional: Save results to disk
    # result.save(filename="result.jpg")
