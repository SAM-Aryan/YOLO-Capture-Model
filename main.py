from ultralytics import YOLO

def main():
  model  = YOLO("yolov8n-seg.pt")
  model.to('cuda')
  model.predict(source=0, show=True)

main()