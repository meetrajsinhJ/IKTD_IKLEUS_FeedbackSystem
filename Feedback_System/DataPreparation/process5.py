
import os
import pandas as pd
import shutil
from pathlib import Path
import json


class CompleteDatasetMapper:
    def __init__(self, csv_folder, images_folder, output_folder):
        """
        Initialize the mapper with necessary paths.

        Args:
            csv_folder (str): Path to folder containing individual CSV files
            images_folder (str): Path to folder containing images
            output_folder (str): Path to save the organized dataset
        """
        self.csv_folder = Path(csv_folder)
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)

        # Create output directory structure
        self.organized_images_dir = self.output_folder / "images"
        self.organized_annotations_dir = self.output_folder / "annotations"

        # Create directories
        self.organized_images_dir.mkdir(parents=True, exist_ok=True)
        self.organized_annotations_dir.mkdir(parents=True, exist_ok=True)

    def find_image_file(self, file_id):
        """
        Find the corresponding image file for a given file_id.
        Checks for common image extensions.
        """
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        for ext in extensions:
            image_path = self.images_folder / f"{file_id}{ext}"
            if image_path.exists():
                return image_path
        return None

    def create_organized_dataset(self):
        """
        Create an organized dataset with images and their annotations.
        """
        # Initialize counters and lists for tracking
        processed_images = 0
        skipped_images = 0
        missing_images = []
        dataset_info = []
        total_annotations = 0

        print("Starting dataset organization...")

        # Process each CSV file in the folder
        for csv_file in self.csv_folder.glob('*_mapped.csv'):
            try:
                # Read the CSV file
                file_annotations = pd.read_csv(csv_file)

                # Extract file_id from CSV filename
                file_id = csv_file.stem.replace('_mapped', '')

                # Find corresponding image
                image_path = self.find_image_file(file_id)

                if image_path is None:
                    print(f"Warning: No image found for file_id: {file_id}")
                    missing_images.append(file_id)
                    continue

                # Create file-specific annotation data
                annotation_data = {
                    'image_id': file_id,
                    'image_filename': image_path.name,
                    'annotations': []
                }

                # Process each annotation for this image
                for _, row in file_annotations.iterrows():
                    annotation = {
                        'row_number': int(row['row_number']),
                        'bbox': {
                            'x_center': float(row['x_center']),
                            'y_center': float(row['y_center']),
                            'width': float(row['width']),
                            'height': float(row['height'])
                        },
                        'annotation_text': row['annotation_text']
                    }
                    annotation_data['annotations'].append(annotation)
                    total_annotations += 1

                # Copy image to organized folder
                new_image_path = self.organized_images_dir / image_path.name
                shutil.copy2(image_path, new_image_path)

                # Save annotation data as JSON
                annotation_file = self.organized_annotations_dir / f"{file_id}_annotations.json"
                with open(annotation_file, 'w', encoding='utf-8') as f:
                    json.dump(annotation_data, f, ensure_ascii=False, indent=2)

                dataset_info.append(annotation_data)
                processed_images += 1
                print(f"Processed {file_id}: Image and annotations organized successfully")

            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                skipped_images += 1

        # Save complete dataset info
        dataset_file = self.output_folder / "complete_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_info': {
                    'total_images': processed_images,
                    'total_annotations': total_annotations
                },
                'images': dataset_info
            }, f, ensure_ascii=False, indent=2)

        # Save processing summary
        summary_file = self.output_folder / "processing_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Dataset Processing Summary\n")
            f.write("========================\n")
            f.write(f"Total processed images: {processed_images}\n")
            f.write(f"Total skipped images: {skipped_images}\n")
            f.write(f"Total annotations: {total_annotations}\n")
            f.write(f"Missing images: {', '.join(missing_images)}\n")

        print("\nDataset Organization Complete!")
        print(f"Processed images: {processed_images}")
        print(f"Skipped images: {skipped_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"\nOrganized dataset saved to: {self.output_folder}")

        return dataset_file


# Example usage
if __name__ == "__main__":
    # Define paths
    csv_folder = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/output"  # Folder containing individual CSV files
    images_folder = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected"  # Folder containing the images
    output_folder = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output"  # Where to save the organized dataset

    # Create and run the mapper
    mapper = CompleteDatasetMapper(csv_folder, images_folder, output_folder)
    dataset_file = mapper.create_organized_dataset()
