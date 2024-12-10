import os
import shutil


def merge_directories(directories, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a counter for sequential naming
    image_counter = 1

    # Loop through each input directory
    for directory in directories:
        # List all files in the directory
        files = os.listdir(directory)

        for file in files:
            # Build the full file path
            src_path = os.path.join(directory, file)

            # Check if it's a file (and optionally if it's an image)
            if os.path.isfile(src_path):
                # Generate the new filename with sequential numbering
                file_extension = os.path.splitext(file)[1]
                new_filename = f"{image_counter}{file_extension}"

                # Destination path
                dst_path = os.path.join(output_dir, new_filename)

                # Copy the file to the new location
                shutil.copy(src_path, dst_path)

                # Increment the counter
                image_counter += 1

    print(f"All images have been merged into '{output_dir}'.")


# Directories to merge
input_directories = [
    "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/dataset/annotations",
    "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/dataset/annotations",
    "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/dataset/annotations",
    "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/dataset/annotations"
]

# Output directory
output_directory = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/dataset/annotationsBB"

# Run the function
merge_directories(input_directories, output_directory)
