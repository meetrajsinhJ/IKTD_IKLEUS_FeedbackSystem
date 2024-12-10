import os
from PIL import Image


def is_image_file(filename):
    # Check if a file is an image based on its extension
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return filename.lower().endswith(image_extensions)


def organize_images_numerically(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter out only image files
    image_files = [f for f in files if is_image_file(f)]

    # Sort image files (this can be adjusted if you want a specific order)
    image_files.sort()

    # Rename files numerically
    for index, filename in enumerate(image_files, start=1):
        # Generate new file name with numerical sequence
        new_filename = f"{index:03d}{os.path.splitext(filename)[1]}"
        old_file_path = os.path.join(directory_path, filename)
        new_file_path = os.path.join(directory_path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")


# Example usage
directory_path = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Dataset/drawing_evaluationTable/tables"
organize_images_numerically(directory_path)
