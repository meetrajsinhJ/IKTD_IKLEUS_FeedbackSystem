import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
from dataclasses import dataclass
import logging


@dataclass
class AnnotationRegion:
    """Data class to store annotation region information"""
    row_number: int
    bbox: Dict[str, float]
    text: str
    image_height: int
    image_width: int

    def get_pixel_coordinates(self) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates"""
        x_center = int(self.bbox['x_center'] * self.image_width)
        y_center = int(self.bbox['y_center'] * self.image_height)
        width = int(self.bbox['width'] * self.image_width)
        height = int(self.bbox['height'] * self.image_height)

        x1 = max(0, x_center - width // 2)
        y1 = max(0, y_center - height // 2)
        x2 = min(self.image_width, x_center + width // 2)
        y2 = min(self.image_height, y_center + height // 2)

        return x1, y1, x2, y2


@dataclass
class ProcessedDrawing:
    """Data class to store processed drawing information"""
    original_image: np.ndarray
    preprocessed_image: np.ndarray
    binary_image: np.ndarray
    detected_lines: np.ndarray
    detected_contours: List[np.ndarray]
    features: Dict
    metadata: Dict
    annotations: List[AnnotationRegion]
    annotation_regions: Dict[int, np.ndarray]


class TechnicalDrawingProcessor:
    """Process technical drawings for ML model input"""

    def __init__(self,
                 target_size: Tuple[int, int] = (800, 800),
                 binary_threshold: int = 127,
                 gaussian_kernel: Tuple[int, int] = (5, 5)):
        self.target_size = target_size
        self.binary_threshold = binary_threshold
        self.gaussian_kernel = gaussian_kernel

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize while maintaining aspect ratio
        aspect_ratio = gray.shape[1] / gray.shape[0]
        if aspect_ratio > 1:
            new_width = self.target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.target_size[1]
            new_width = int(new_height * aspect_ratio)

        resized = cv2.resize(gray, (new_width, new_height))

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, self.gaussian_kernel, 0)

        return blurred

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        image = self.load_image(image_path)
        return self.preprocess_image(image)

    def create_binary_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image using adaptive thresholding"""
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        return binary

    def detect_lines(self, image: np.ndarray) -> np.ndarray:
        """Detect lines using Hough transform"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=100,
            maxLineGap=10
        )
        return lines if lines is not None else np.array([])

    def detect_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect contours in the drawing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def extract_features(self, image: np.ndarray) -> Dict:
        """Extract features from the image"""
        features = {
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),

            # Add more sophisticated features here
            'histogram': cv2.calcHist(
                [image],
                [0],
                None,
                [256],
                [0, 256]
            ).flatten().tolist(),

            # Edge features
            'edge_density': np.mean(cv2.Canny(image, 50, 150)) / 255.0
        }
        return features

    def process_annotations(self,
                            annotations_path: str,
                            image: np.ndarray) -> Tuple[List[AnnotationRegion], Dict[int, np.ndarray]]:
        """Process annotations and extract regions"""
        height, width = image.shape[:2]

        # Load annotations
        with open(annotations_path, 'r') as f:
            data = json.load(f)

        annotations = []
        regions = {}

        # Process each annotation
        for ann in data.get('annotations', []):
            annotation = AnnotationRegion(
                row_number=ann['row_number'],
                bbox=ann['bbox'],
                text=ann['annotation_text'],
                image_height=height,
                image_width=width
            )

            # Extract region
            x1, y1, x2, y2 = annotation.get_pixel_coordinates()
            region = image[y1:y2, x1:x2]

            annotations.append(annotation)
            regions[ann['row_number']] = region

        return annotations, regions

    def process_drawing_pair(self,
                             incorrect_path: str,
                             correct_path: str,
                             annotations_path: Optional[str] = None) -> Tuple[ProcessedDrawing, ProcessedDrawing]:
        """Process a pair of drawings with annotations"""
        # Load and preprocess images
        incorrect_img = self.load_and_preprocess(incorrect_path)
        correct_img = self.load_and_preprocess(correct_path)

        # Process annotations if provided
        annotations = []
        annotation_regions = {}
        if annotations_path:
            annotations, annotation_regions = self.process_annotations(
                annotations_path,
                incorrect_img
            )

        # Create ProcessedDrawing objects
        incorrect_processed = ProcessedDrawing(
            original_image=incorrect_img,
            preprocessed_image=incorrect_img,  # Already preprocessed
            binary_image=self.create_binary_image(incorrect_img),
            detected_lines=self.detect_lines(incorrect_img),
            detected_contours=self.detect_contours(incorrect_img),
            features=self.extract_features(incorrect_img),
            metadata={'type': 'incorrect', 'path': incorrect_path},
            annotations=annotations,
            annotation_regions=annotation_regions
        )

        correct_processed = ProcessedDrawing(
            original_image=correct_img,
            preprocessed_image=correct_img,  # Already preprocessed
            binary_image=self.create_binary_image(correct_img),
            detected_lines=self.detect_lines(correct_img),
            detected_contours=self.detect_contours(correct_img),
            features=self.extract_features(correct_img),
            metadata={'type': 'correct', 'path': correct_path},
            annotations=[],
            annotation_regions={}
        )

        return incorrect_processed, correct_processed

    def visualize_annotations(self, drawing: ProcessedDrawing, output_path: str):
        """Visualize annotations on the drawing with improved visibility"""
        # Ensure we're working with the original image
        if len(drawing.original_image.shape) == 2:
            vis_image = cv2.cvtColor(drawing.original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = drawing.original_image.copy()

        # Draw each annotation with different colors for better visibility
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255)  # Magenta
        ]

        print(f"\nVisualizing {len(drawing.annotations)} annotations:")

        for idx, ann in enumerate(drawing.annotations):
            try:
                x1, y1, x2, y2 = ann.get_pixel_coordinates()
                color = colors[idx % len(colors)]

                # Draw filled rectangle with transparency
                overlay = vis_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0, vis_image)

                # Add annotation text with background for better visibility
                text = f"Region {ann.row_number}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                # Draw text background
                cv2.rectangle(vis_image,
                              (x1, y1 - text_height - 10),
                              (x1 + text_width + 10, y1),
                              (255, 255, 255),
                              -1)

                # Draw text
                cv2.putText(vis_image,
                            text,
                            (x1 + 5, y1 - 5),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness)

                print(f"Drew annotation {ann.row_number} at coordinates ({x1}, {y1}, {x2}, {y2})")

            except Exception as e:
                print(f"Error drawing annotation {idx}: {str(e)}")

        # Save the visualization
        try:
            success = cv2.imwrite(output_path, vis_image)
            if success:
                print(f"\nSuccessfully saved visualization to {output_path}")
            else:
                print(f"\nFailed to save visualization to {output_path}")
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")


def main():
    """Example usage with improved output"""
    processor = TechnicalDrawingProcessor()

    try:
        # Get current directory
        current_dir = Path.cwd()

        # Process drawings with annotations
        incorrect_processed, correct_processed = processor.process_drawing_pair(
            str(current_dir / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/1.jpg"),
            str(current_dir / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/1.jpg"),
            str(current_dir / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/1_annotations.json")
        )

        # Create output directory if it doesn't exist
        output_dir = current_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Visualize annotations
        output_path = output_dir / "annotated_drawing.png"
        processor.visualize_annotations(incorrect_processed, str(output_path))

        # Print detailed results
        print("\nProcessing Results:")
        print("-" * 50)
        print(f"Number of annotations: {len(incorrect_processed.annotations)}")
        print(f"Number of detected lines: {len(incorrect_processed.detected_lines)}")
        print(f"Number of contours: {len(incorrect_processed.detected_contours)}")

        print("\nAnnotation Details:")
        print("-" * 50)
        for ann in incorrect_processed.annotations:
            print(f"\nRegion {ann.row_number}:")
            print(f"Text: {ann.text}")
            x1, y1, x2, y2 = ann.get_pixel_coordinates()
            print(f"Coordinates: ({x1}, {y1}) to ({x2}, {y2})")

            # Get region features
            region = incorrect_processed.annotation_regions.get(ann.row_number)
            if region is not None:
                print(f"Region size: {region.shape}")

        print("\nImage Features:")
        print("-" * 50)
        for key, value in incorrect_processed.features.items():
            if key != 'histogram':  # Skip printing the long histogram
                print(f"{key}: {value}")

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()