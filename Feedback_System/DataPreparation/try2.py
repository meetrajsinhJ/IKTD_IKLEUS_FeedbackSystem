import os
import pandas as pd


def parse_bbox_file(file_path):
    """Parse bounding box coordinates file."""
    bboxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    category_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    bboxes.append({
                        'category_id': category_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    except Exception as e:
        print(f"Error parsing bbox file {file_path}: {str(e)}")
        return []
    return bboxes


def parse_text_file(file_path):
    """Parse text annotation file."""
    annotations = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_id = None
            current_text = []

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split by first hyphen
                parts = line.split('-', 1)

                # Try to extract category ID from the start of the line
                try:
                    potential_id = int(parts[0].split()[0])
                    if current_id is not None and current_text:
                        annotations[current_id] = ' '.join(current_text)
                    current_id = potential_id
                    current_text = [parts[1]] if len(parts) > 1 else []
                except ValueError:
                    # If line starts with non-number, append to current text
                    if current_id is not None:
                        current_text.append(line)

            # Don't forget to add the last annotation
            if current_id is not None and current_text:
                annotations[current_id] = ' '.join(current_text)

    except Exception as e:
        print(f"Error parsing text file {file_path}: {str(e)}")
        return {}
    return annotations


def create_mapping(bbox_dir, text_dir, output_file='drawing_annotations.csv'):
    """Create mapping between bounding boxes and text annotations."""
    all_data = []

    # Get list of bbox files
    bbox_files = [f for f in os.listdir(bbox_dir) if f.endswith('.txt')]

    for bbox_file in bbox_files:
        # Get corresponding text file name
        base_name = bbox_file.split('.')[0]
        text_file = f"{base_name}_name.txt"
        text_file_path = os.path.join(text_dir, text_file)
        bbox_file_path = os.path.join(bbox_dir, bbox_file)

        # Check if both files exist
        if not os.path.exists(text_file_path):
            print(f"Warning: No corresponding text file found for {bbox_file}")
            continue

        # Parse both files
        bboxes = parse_bbox_file(bbox_file_path)
        text_annotations = parse_text_file(text_file_path)

        # Combine the data
        for bbox in bboxes:
            category_id = bbox['category_id']
            annotation_text = text_annotations.get(category_id, '')

            all_data.append({
                'file_id': base_name,
                'category_id': category_id,
                'x_center': bbox['x_center'],
                'y_center': bbox['y_center'],
                'width': bbox['width'],
                'height': bbox['height'],
                'annotation_text': annotation_text
            })

    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully created mapping CSV file: {output_file}")
        print(f"Total mapped annotations: {len(df)}")

        # Print summary of unique file IDs processed
        print(f"Files processed: {len(df['file_id'].unique())}")
    else:
        print("No data was processed. Check your input directories and files.")


# Example usage
if __name__ == "__main__":
    bbox_directory = "/Users/jadeja19/Documents/Hiwi iktd/new_data/IKILeUS_Trainingsdaten/01_Schraubenzeichnung_annotiert/a01/txt_files"  # Directory containing bbox files
    text_directory = "/Users/jadeja19/Documents/Hiwi iktd/new_data/IKILeUS_Trainingsdaten/01_Schraubenzeichnung_annotiert/a01/name_txt_files"  # Directory containing text annotation files

    create_mapping(
        bbox_dir=bbox_directory,
        text_dir=text_directory,
        output_file='drawing_annotations.csv'
    )