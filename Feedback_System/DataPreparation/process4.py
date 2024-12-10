
import os
import pandas as pd
from pathlib import Path


class BatchAnnotationMapper:
    def __init__(self, bb_folder, annotation_folder, output_folder):
        """
        Initialize the mapper with source and destination folders.

        Args:
            bb_folder (str): Path to folder containing bounding box files
            annotation_folder (str): Path to folder containing text annotation files
            output_folder (str): Path to folder where CSV files will be saved
        """
        self.bb_folder = Path(bb_folder)
        self.annotation_folder = Path(annotation_folder)
        self.output_folder = Path(output_folder)

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def process_single_pair(self, bb_file, annotation_file):
        """
        Process a single pair of bounding box and annotation files.

        Args:
            bb_file (Path): Path to bounding box file
            annotation_file (Path): Path to annotation file

        Returns:
            pandas.DataFrame: DataFrame containing the mapped data
        """
        # Read bounding box coordinates
        bboxes = []
        with open(bb_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append({
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })

        # Read text annotations
        texts = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split('-', 1)
                    if len(parts) > 1:
                        texts.append(parts[1].strip())
                    else:
                        texts.append(line.strip())

        # Combine the data sequentially
        combined_data = []
        for i, (bbox, text) in enumerate(zip(bboxes, texts)):
            combined_data.append({
                'file_id': bb_file.stem,
                'row_number': i + 1,
                'x_center': bbox['x_center'],
                'y_center': bbox['y_center'],
                'width': bbox['width'],
                'height': bbox['height'],
                'annotation_text': text
            })

        return pd.DataFrame(combined_data)

    def process_all_files(self):
        """
        Process all matching pairs of files in the source folders.
        """
        # Keep track of processing statistics
        processed_pairs = 0
        failed_pairs = 0
        all_data = []

        # Process each bounding box file
        for bb_file in self.bb_folder.glob('*.txt'):
            # Find corresponding annotation file
            annotation_file = self.annotation_folder / f"{bb_file.stem}_name.txt"

            if not annotation_file.exists():
                print(f"Warning: No matching annotation file for {bb_file.name}")
                failed_pairs += 1
                continue

            try:
                print(f"Processing pair: {bb_file.name} and {annotation_file.name}")
                df = self.process_single_pair(bb_file, annotation_file)

                # Save individual CSV file
                output_file = self.output_folder / f"{bb_file.stem}_mapped.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')

                # Add to combined data
                all_data.append(df)
                processed_pairs += 1

            except Exception as e:
                print(f"Error processing {bb_file.name}: {str(e)}")
                failed_pairs += 1

        # Create combined CSV file if there's data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_output = self.output_folder / "all_annotations_combined.csv"
            combined_df.to_csv(combined_output, index=False, encoding='utf-8')

            # Print summary statistics
            print("\nProcessing Summary:")
            print(f"Successfully processed pairs: {processed_pairs}")
            print(f"Failed pairs: {failed_pairs}")
            print(f"Total annotations: {len(combined_df)}")
            print(f"\nOutput files saved to: {self.output_folder}")
            print(f"Combined CSV file: {combined_output}")

            return combined_df

        return None


# Example usage
if __name__ == "__main__":
    # Define your folders
    bb_folder = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/txt_files"  # Folder containing the .txt files
    annotation_folder = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/name_txt_files"  # Folder containing the _name.txt files
    output_folder = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/output"  # Where to save the CSV files

    # Create and run the mapper
    mapper = BatchAnnotationMapper(bb_folder, annotation_folder, output_folder)
    combined_dataset = mapper.process_all_files()
