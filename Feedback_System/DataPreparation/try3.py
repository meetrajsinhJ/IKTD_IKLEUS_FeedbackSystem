import pandas as pd


def create_sequential_mapping_csv(bbox_file, text_file, output_csv):
    """
    Create a CSV mapping bounding box coordinates with text annotations based on row sequence.

    Args:
        bbox_file (str): Path to the bounding box coordinates file (1.txt)
        text_file (str): Path to the text annotations file (1_name.txt)
        output_csv (str): Path for the output CSV file
    """
    # Read bounding box coordinates
    bboxes = []
    with open(bbox_file, 'r') as f:
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
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Split by first hyphen and take everything after it
                parts = line.split('-', 1)
                if len(parts) > 1:
                    texts.append(parts[1].strip())
                else:
                    texts.append(line.strip())

    # Combine the data sequentially
    combined_data = []
    for i, (bbox, text) in enumerate(zip(bboxes, texts)):
        combined_data.append({
            'row_number': i + 1,  # 1-based indexing for readability
            'x_center': bbox['x_center'],
            'y_center': bbox['y_center'],
            'width': bbox['width'],
            'height': bbox['height'],
            'annotation_text': text
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(combined_data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    # Print summary
    print("\nFirst few rows of the created CSV:")
    print(df.head())
    print(f"\nCSV file saved as: {output_csv}")
    print(f"Total mapped annotations: {len(df)}")

    # Verify row counts match
    print(f"\nNumber of bounding boxes: {len(bboxes)}")
    print(f"Number of text annotations: {len(texts)}")
    if len(bboxes) != len(texts):
        print("Warning: Number of bounding boxes and text annotations don't match!")


# Example usage
bbox_file = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/txt_files/2.txt"
text_file = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/name_txt_files/2_name.txt"
output_csv = "drawing_annotations_sequential2.csv"

create_sequential_mapping_csv(bbox_file, text_file, output_csv)