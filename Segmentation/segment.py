import os
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Segmentation/segmentmodel/train/weights/best.pt")

# Define the directory containing images
image_dir = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/images"
output_dir = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/output_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Collect all image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

# Process each image and save the results
for idx, image_path in enumerate(image_paths):
    # Read and display the image
    image = cv2.imread(image_path)
    cv2.imshow('Original Image', image)
    print(f"Press any key to process {image_path}")

    # Wait for a key press to move to the next image
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Destroy the displayed window

    # Perform inference
    results = model(image_path)  # Perform inference on the image

    # Process and display results
    result = results[0]
    result.show()  # Display the result
    output_path = os.path.join(output_dir, f"result_{idx}.jpg")
    result.save(filename=output_path)  # Save to disk

    print(f"Processed and saved result for {image_path} as {output_path}")

# Final cleanup: destroy all windows
cv2.destroyAllWindows()
print("All images processed and windows destroyed.")
