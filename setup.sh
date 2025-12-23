mkdir -p models 
pushd models
for model in yolo11n yolo11s yolo11m yolo11l yolo11x; do
  if [ ! -f ${model}.pt ]; then
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/${model}.pt
  fi
  for type in seg cls pose obb; do
    if [ ! -f ${model}-${type}.pt ]; then
      wget https://github.com/ultralytics/assets/releases/download/v8.3.0/${model}-${type}.pt
    fi
  done
done
if [ ! -f coco.yaml ]; then
  wget https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/ultralytics/cfg/datasets/coco.yaml
fi
if [ ! -f ImageNet.yaml ]; then
  wget https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/ultralytics/cfg/datasets/ImageNet.yaml
fi
popd
