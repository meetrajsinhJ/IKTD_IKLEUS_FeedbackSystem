import albumentations as A
import cv2
import os
import numpy as np

# Define your augmentation pipeline
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.5),  # Translation
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Noise Injection
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness & Contrast
    # For erosion and dilation, you can use more advanced morphological operations if needed
    A.ElasticTransform(alpha=2, sigma=50, alpha_affine=50, p=0.5)  # Simulating Erosion & Dilation effects
])


# Function to check if a file is an image based on its extension
def is_image_file(filename):
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    return os.path.splitext(filename)[1].lower() in valid_image_extensions


# Function to process and save an augmented image
def augment_and_save_image(image_path, save_dir):
    # Read the image
    image = cv2.imread(image_path)

    # Ensure the image is read correctly
    if image is None:
        print(f"Error reading image {image_path}. It might be corrupted or the path could be incorrect.")
        return

    # Ensure the image is in the correct color format (albumentations expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the transformations
    augmented_image = transform(image=image)['image']

    # Convert the image back to BGR color format before saving (cv2 expects BGR)
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

    # Construct the save path
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"aug_{base_name}")

    # Save the augmented image
    cv2.imwrite(save_path, augmented_image)


# List of your directories
directories = [
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/01Schraubedrawcorrected',
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/02corrected',
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/03corrected',
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/04corrected',
    '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/05corrected'
]

# Output directory for augmented images
output_base_dir = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/alldataaugtech'

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Process all directories
for dataset_dir in directories:
    # Create an output directory for each input directory
    output_dir = os.path.join(output_base_dir, os.path.basename(dataset_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all files in the directory, only if they are image files
    for file_name in os.listdir(dataset_dir):
        full_image_path = os.path.join(dataset_dir, file_name)
        if os.path.isfile(full_image_path) and is_image_file(file_name):
            augment_and_save_image(full_image_path, output_dir)
        else:
            print(f"Skipping non-image file: {full_image_path}")
