import os
import shutil


def organize_files(source_dir, txt_dir="txt_files", name_txt_dir="name_txt_files"):
    """
    Organize files from source directory into separate folders for .txt and name.txt files.

    Args:
        source_dir (str): Path to the source directory containing the files
        txt_dir (str): Name of the directory for regular .txt files
        name_txt_dir (str): Name of the directory for name.txt files
    """
    # Create destination directories if they don't exist
    txt_path = os.path.join(source_dir, txt_dir)
    name_txt_path = os.path.join(source_dir, name_txt_dir)

    os.makedirs(txt_path, exist_ok=True)
    os.makedirs(name_txt_path, exist_ok=True)

    # Get list of files in source directory
    files = os.listdir(source_dir)

    # Counter for moved files
    moved_files = {"txt": 0, "name_txt": 0}

    # Process each file
    for file in files:
        # Skip if it's a directory
        if os.path.isdir(os.path.join(source_dir, file)):
            continue

        # Process only .txt files
        if file.endswith('.txt'):
            source_file = os.path.join(source_dir, file)

            # Determine destination based on filename
            if 'name' in file:
                dest = os.path.join(name_txt_path, file)
                moved_files["name_txt"] += 1
            else:
                dest = os.path.join(txt_path, file)
                moved_files["txt"] += 1

            # Move the file
            try:
                shutil.move(source_file, dest)
                print(f"Moved {file} to {'name_txt_files' if 'name' in file else 'txt_files'}")
            except Exception as e:
                print(f"Error moving {file}: {str(e)}")

    # Print summary
    print("\nOrganization Complete!")
    print(f"Regular .txt files moved: {moved_files['txt']}")
    print(f"Name.txt files moved: {moved_files['name_txt']}")


# Example usage
if __name__ == "__main__":
    # Replace this with your actual directory path
    source_directory = "/Users/jadeja19/Documents/Hiwi iktd/new_data/IKILeUS_Trainingsdaten/01_Schraubenzeichnung_annotiert/a01"

    # Run the organization
    organize_files(source_directory)