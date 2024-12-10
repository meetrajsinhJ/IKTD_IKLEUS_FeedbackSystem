import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from collections import defaultdict
import json
import logging
from pathlib import Path


@dataclass
class FeedbackItem:
    """Data class for storing feedback information"""
    criterion_id: int
    main_category: str
    sub_category: str
    measurements: List[Dict[str, float]]
    is_verified: bool  # indicates if marked as "Sicher"
    raw_text: str
    bbox: Optional[Dict[str, float]] = None  # Adding bounding box information


class AnnotationTextProcessor:
    """Process technical drawing annotation text with minimal dependencies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Measurement patterns
        self.patterns = {
            'diameter': r'(?P<type>d[b2]?)\s*=\s*(?P<value>\d+(?:[,.]\d+)?)\s*(?:mm)?',
            'length': r'(?P<value>\d+(?:[,.]\d+)?)\s*(?:mm)',
            'angle': r'(?P<value>\d+(?:[,.]\d+)?)\s*(?:°|grad)',
            'thread': r'M(?P<major>\d+)\s*x\s*(?P<pitch>\d+)',
            'range': r'(?P<min>\d+(?:[,.]\d+)?)\s*\.{2,3}\s*(?P<max>\d+(?:[,.]\d+)?)',
            'ratio': r'(?P<num>\d+)\s*[:/]\s*(?P<den>\d+)'
        }

    def load_annotation_file(self, file_path: str) -> Dict:
        """Load annotations from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Error loading annotation file {file_path}: {str(e)}")
            return {}

    def extract_measurements(self, text: str) -> List[Dict]:
        """Extract measurements from text using regex patterns"""
        measurements = []

        for measure_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                measurement = {'type': measure_type}
                # Convert matched values to float, replacing comma with dot
                for key, value in match.groupdict().items():
                    if value:
                        try:
                            measurement[key] = float(value.replace(',', '.'))
                        except ValueError:
                            measurement[key] = value
                measurements.append(measurement)

        return measurements

    def parse_annotation_text(self, annotation: Dict) -> Optional[FeedbackItem]:
        """Parse a single annotation entry"""
        try:
            text = annotation.get('annotation_text', '')

            # Split by hyphen to separate parts
            parts = [p.strip() for p in text.split('-') if p.strip()]
            if not parts:
                return None

            # Extract criterion ID from the first part
            criterion_match = re.match(r'(\d+(?:\.\d+)?)\s*(.*)', parts[0])
            if not criterion_match:
                return None

            criterion_id = float(criterion_match.group(1))
            main_category = criterion_match.group(2).strip()

            # Get subcategory from remaining parts
            sub_category = parts[1] if len(parts) > 1 else ""

            # Check if marked as verified ("Sicher")
            is_verified = any("Sicher" in part for part in parts)

            # Extract measurements
            measurements = self.extract_measurements(text)

            return FeedbackItem(
                criterion_id=int(criterion_id),
                main_category=main_category,
                sub_category=sub_category,
                measurements=measurements,
                is_verified=is_verified,
                raw_text=text,
                bbox=annotation.get('bbox')
            )

        except Exception as e:
            self.logger.error(f"Error parsing annotation: {str(e)}")
            return None

    def process_annotations(self, annotations_data: Dict) -> List[FeedbackItem]:
        """Process all annotations from the file"""
        feedback_items = []

        for annotation in annotations_data.get('annotations', []):
            item = self.parse_annotation_text(annotation)
            if item:
                feedback_items.append(item)

        return feedback_items

    def analyze_feedback_patterns(self, feedback_items: List[FeedbackItem]) -> Dict:
        """Analyze patterns in feedback items"""
        analysis = {
            'criterion_counts': defaultdict(int),
            'measurement_types': defaultdict(int),
            'verified_items': 0,
            'total_items': len(feedback_items),
            'measurements_by_criterion': defaultdict(list),
            'spatial_distribution': defaultdict(list)  # Adding spatial analysis
        }

        for item in feedback_items:
            analysis['criterion_counts'][item.criterion_id] += 1

            if item.is_verified:
                analysis['verified_items'] += 1

            for measurement in item.measurements:
                analysis['measurement_types'][measurement['type']] += 1
                analysis['measurements_by_criterion'][item.criterion_id].append(
                    measurement
                )

            # Add spatial information if available
            if item.bbox:
                analysis['spatial_distribution'][item.criterion_id].append(item.bbox)

        return analysis


def process_multiple_annotation_files(file_paths: List[str]) -> Dict[str, List[FeedbackItem]]:
    """Process multiple annotation files and combine results"""
    processor = AnnotationTextProcessor()
    all_results = {}

    for file_path in file_paths:
        try:
            # Load and process annotations
            annotations_data = processor.load_annotation_file(file_path)
            feedback_items = processor.process_annotations(annotations_data)

            # Store results with file name as key
            file_name = Path(file_path).stem
            all_results[file_name] = feedback_items

            print(f"\nProcessed {file_name}:")
            print(f"Found {len(feedback_items)} feedback items")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return all_results


def main():
    """Example usage with annotation files"""

    # List of annotation files to process
    annotation_files = [
        "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/1_annotations.json",
        "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/2_annotations.json",
        "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/3_annotations.json",
        "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/4_annotations.json"
    ]

    # Process all annotation files
    results = process_multiple_annotation_files(annotation_files)

    # Analyze results for each file
    for file_name, feedback_items in results.items():
        print(f"\nAnalysis for {file_name}:")
        print("-" * 50)

        # Print feedback items
        for item in feedback_items:
            print(f"\nCriterion {item.criterion_id}:")
            print(f"Main Category: {item.main_category}")
            print(f"Sub Category: {item.sub_category}")
            print(f"Measurements: {item.measurements}")
            print(f"Verified: {item.is_verified}")
            print(f"Location: {item.bbox}")

        # Get pattern analysis
        processor = AnnotationTextProcessor()
        analysis = processor.analyze_feedback_patterns(feedback_items)

        print("\nFeedback Analysis:")
        print(f"Total items: {analysis['total_items']}")
        print(f"Verified items: {analysis['verified_items']}")

        print("\nCriterion counts:")
        for criterion, count in analysis['criterion_counts'].items():
            print(f"Criterion {criterion}: {count} items")

        print("\nMeasurement types:")
        for mtype, count in analysis['measurement_types'].items():
            print(f"{mtype}: {count} occurrences")

        print("\nSpatial Distribution:")
        for criterion, locations in analysis['spatial_distribution'].items():
            print(f"Criterion {criterion}: {len(locations)} locations")


if __name__ == "__main__":
    main()