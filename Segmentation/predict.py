import cv2
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Segmentation/detectionmodel/weights/best.pt")

# Define the directory containing images
image_dir = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/images"
output_dir = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/output_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Define function to perform object detection on each image in the directory
def detect_objects_in_directory(image_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image {image_path}")
                continue

            # Perform inference
            results = model(image)
            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                classes = result.boxes.cls

                for box, confidence, class_id in zip(boxes, confidences, classes):
                    if confidence > 0.5:  # Adjust threshold as needed
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                        # Draw bounding box and label on the image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{model.names[int(class_id)]} {confidence:.2f}"
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the resulting image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)

            # Display the resulting image
            cv2.imshow('Object Detection', image)
            cv2.waitKey(0)  # Wait for a key press to move to the next image
            cv2.destroyAllWindows()

            print(f"Processed {filename}")


# Run the detection on the directory
detect_objects_in_directory(image_dir, output_dir)
