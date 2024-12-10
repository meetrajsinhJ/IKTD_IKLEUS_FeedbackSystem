from ultralytics import YOLO

# Load a model

model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data='/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Segmentation/iktd.v1i.yolov8/data.yaml',
                      epochs= 25,
                      imgsz=640,
                      project='/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Segmentation/segmentmodel',  # Directory to save the runs
                      )  # Subdirectory for this specific run
