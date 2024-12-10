import torch
import numpy as np
from pathlib import Path
import logging
import json
import cv2
import datetime
from typing import Dict, List
from dataclasses import dataclass

# Import from your modules
from LVmodule.Multifusion.imageProcessing import TechnicalDrawingProcessor
from LVmodule.Multifusion.mainModel import MultiFusionModel


class SingleImageInference:
    """Class for performing inference on a single technical drawing"""

    def __init__(self, model_path: str, device: str = None):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Set device
        self.device = device if device else \
            ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize image processor
        self.image_processor = TechnicalDrawingProcessor()

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str) -> MultiFusionModel:
        """Load the trained model"""
        try:
            # Initialize model architecture
            model = MultiFusionModel(
                visual_dim=256,
                text_dim=384,
                hidden_dim=256,
                num_criteria=9,
                max_seq_length=10
            )

            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            self.logger.info(f"Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_review_criteria(self) -> List[Dict]:
        """Load the original review criteria"""
        return [
            {
                "nr": 1,
                "lernziel": "Schnitt-darstellung",
                "stichwort": "Schraube allgemein",
                "kriterium": "Länge l = 30 mm, Durchmesser d1 = 40 mm Fase am unteren Ende (zwei umlaufende Körperkanten) Abstand Gewinde - Kopf max. 3·P = 15 mm"
            },
            {
                "nr": 2,
                "lernziel": "Schrauben-darstellung",
                "stichwort": "Schraubenkopf",
                "kriterium": "Tangentialer Übergang, Höhe k = 40 mm (± 1 mm)"
            },
            {
                "nr": 3,
                "lernziel": "Norm-angaben",
                "stichwort": "Scheibe",
                "kriterium": "Sechskant mit Strichlinie oder nicht dargestellt, Außenkante des Schraubenkopfs gerundet oder gefast (Maß 4 mm)"
            },
            {
                "nr": 4,
                "lernziel": "Schnitt-darstellung",
                "stichwort": "Platte",
                "kriterium": "Außendurchmesser korrekt (d2 = 120 mm) Wenn geschnitten: umlaufende Kanten vollständig"
            },
            {
                "nr": 5,
                "lernziel": "Schrauben-darstellung",
                "stichwort": "Gewindereserve",
                "kriterium": "Durchgangsloch: Durchmesser d3 = 48 mm, zwei Fasen 2x45° (umlaufende Kanten!) X = 3·P = 15 mm"
            },
            {
                "nr": 6,
                "lernziel": "Norm-angaben",
                "stichwort": "Grundloch",
                "kriterium": "e1 = 22...22,4...23 mm, Fase max. 1x45°, Winkel Bohrerspitze = 118°...120°"
            },
            {
                "nr": 7,
                "lernziel": "Schrauben-darstellung",
                "stichwort": "Gewindedarstellung",
                "kriterium": "Schraube: Körperkante breit, Gewindekern schmal Ende Gewinde: umlaufende Körperkante"
            },
            {
                "nr": 8,
                "lernziel": "Schnitt-darstellung",
                "stichwort": "Schraffur",
                "kriterium": "Gewindereserve: Grundloch breit, Kern schmal Ende Gewinde: umlaufende Körperkante"
            },
            {
                "nr": 9,
                "lernziel": "Vorgabe",
                "stichwort": "Schriftfeld",
                "kriterium": "Gegebene Schraffur korrekt fortgesetzt, schmale Volllinie, Schraffur jeweils bis Körperkante"
            }
        ]

    def preprocess_image(self, image_path: str) -> Dict:
        """Preprocess a single image for inference"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Preprocess using image processor
            preprocessed = self.image_processor.preprocess_image(image)
            binary = self.image_processor.create_binary_image(preprocessed)
            lines = self.image_processor.detect_lines(preprocessed)
            contours = self.image_processor.detect_contours(preprocessed)
            features = self.image_processor.extract_features(preprocessed)

            # Extract visual features
            visual_features = []
            visual_features.extend(self._flatten_dict_values(features))
            visual_features.extend([
                float(len(lines) if lines is not None else 0),
                float(len(contours))
            ])

            # Pad or truncate to expected size
            target_size = 256
            if len(visual_features) > target_size:
                visual_features = visual_features[:target_size]
            else:
                visual_features.extend([0] * (target_size - len(visual_features)))

            # Prepare dummy text features (since we don't have annotations)
            text_features = np.zeros((10, 384))  # 10 is max_seq_length, 384 is text_dim
            attention_mask = np.ones(10)  # All positions attended to

            return {
                'visual_features': torch.tensor(visual_features).float(),
                'text_features': torch.tensor(text_features).float(),
                'attention_mask': torch.tensor(attention_mask).bool()
            }

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def _flatten_dict_values(self, d: Dict) -> List:
        """Recursively flatten dictionary values"""
        flattened = []
        for v in d.values():
            if isinstance(v, dict):
                flattened.extend(self._flatten_dict_values(v))
            elif isinstance(v, (list, np.ndarray)):
                if isinstance(v, np.ndarray) and v.dtype == np.dtype('object'):
                    continue
                flattened.extend([float(x) for x in v])
            elif isinstance(v, (int, float)):
                flattened.append(float(v))
        return flattened

    def predict(self, image_path: str) -> Dict:
        """Run inference on a single image"""
        try:
            # Preprocess image
            features = self.preprocess_image(image_path)

            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}

            # Add batch dimension
            features = {k: v.unsqueeze(0) for k, v in features.items()}

            # Run inference
            with torch.no_grad():
                predictions = self.model(
                    features['visual_features'],
                    features['text_features'],
                    features['attention_mask']
                )

                # Convert predictions to numpy
                predictions = predictions.cpu().numpy()[0]

            # Get criteria
            criteria = self._load_review_criteria()

            # Format results
            results = {
                'predictions': {},
                'metadata': {
                    'model_version': '1.0',
                    'timestamp': str(datetime.datetime.now()),
                    'image_path': image_path
                }
            }

            # Add predictions for each criterion
            for idx, criterion in enumerate(criteria, 1):
                results['predictions'][f'criterion_{idx}'] = {
                    'lernziel': criterion['lernziel'],
                    'stichwort': criterion['stichwort'],
                    'kriterium': criterion['kriterium'],
                    'score': float(predictions[idx - 1]),
                    'label': 'correct' if predictions[idx - 1] > 0.5 else 'incorrect',
                    'confidence': float(abs(predictions[idx - 1] - 0.5) * 2)
                }

            return results

        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise

    def visualize_results(self, image_path: str, results: Dict, output_path: str):
        """Visualize the inference results with enhanced visibility"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            height, width = image.shape[:2]

            # Define colors for each criterion (BGR format) with darker shades
            criterion_colors = {
                'criterion_1': (0, 0, 200),  # Darker Red
                'criterion_2': (0, 180, 0),  # Darker Green
                'criterion_3': (200, 0, 0),  # Darker Blue
                'criterion_4': (0, 140, 255),  # Darker Orange
                'criterion_5': (180, 0, 180),  # Darker Magenta
                'criterion_6': (100, 200, 200),  # Darker Cyan
                'criterion_7': (128, 0, 90),  # Darker Purple
                'criterion_8': (0, 80, 160),  # Darker Brown
                'criterion_9': (130, 90, 0)  # Darker Teal
            }

            # Define regions and their corresponding feedback positions
            regions = {
                'criterion_1': {
                    'box': {'x1': int(0.35 * width), 'y1': int(0.2 * height),
                            'x2': int(0.65 * width), 'y2': int(0.3 * height)},
                    'feedback_pos': {'x': int(0.66 * width), 'y': int(0.25 * height)}
                },
                'criterion_2': {
                    'box': {'x1': int(0.35 * width), 'y1': int(0.1 * height),
                            'x2': int(0.65 * width), 'y2': int(0.2 * height)},
                    'feedback_pos': {'x': int(0.66 * width), 'y': int(0.15 * height)}
                },
                'criterion_3': {
                    'box': {'x1': int(0.35 * width), 'y1': int(0.15 * height),
                            'x2': int(0.65 * width), 'y2': int(0.25 * height)},
                    'feedback_pos': {'x': int(0.05 * width), 'y': int(0.2 * height)}
                },
                'criterion_4': {
                    'box': {'x1': int(0.1 * width), 'y1': int(0.3 * height),
                            'x2': int(0.9 * width), 'y2': int(0.7 * height)},
                    'feedback_pos': {'x': int(0.05 * width), 'y': int(0.4 * height)}
                },
                'criterion_5': {
                    'box': {'x1': int(0.35 * width), 'y1': int(0.4 * height),
                            'x2': int(0.65 * width), 'y2': int(0.5 * height)},
                    'feedback_pos': {'x': int(0.66 * width), 'y': int(0.45 * height)}
                },
                'criterion_6': {
                    'box': {'x1': int(0.35 * width), 'y1': int(0.6 * height),
                            'x2': int(0.65 * width), 'y2': int(0.7 * height)},
                    'feedback_pos': {'x': int(0.66 * width), 'y': int(0.65 * height)}
                },
                'criterion_7': {
                    'box': {'x1': int(0.35 * width), 'y1': int(0.35 * height),
                            'x2': int(0.65 * width), 'y2': int(0.45 * height)},
                    'feedback_pos': {'x': int(0.05 * width), 'y': int(0.6 * height)}
                },
                'criterion_8': {
                    'box': {'x1': int(0.1 * width), 'y1': int(0.3 * height),
                            'x2': int(0.9 * width), 'y2': int(0.7 * height)},
                    'feedback_pos': {'x': int(0.05 * width), 'y': int(0.8 * height)}
                },
                'criterion_9': {
                    'box': {'x1': int(0.1 * width), 'y1': int(0.85 * height),
                            'x2': int(0.9 * width), 'y2': int(0.95 * height)},
                    'feedback_pos': {'x': int(0.05 * width), 'y': int(0.9 * height)}
                }
            }

            # Create a copy for visualization
            vis_image = image.copy()

            # Add title with larger font
            cv2.putText(vis_image,
                        "Technishe Zeichnung Bewertung",
                        (int(0.3 * width), 40),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.2,  # Increased font size
                        (0, 0, 0),
                        3)  # Increased thickness

            # Process each criterion
            for criterion, result in results['predictions'].items():
                if criterion not in regions:
                    continue

                color = criterion_colors[criterion]
                region = regions[criterion]

                # Draw bounding box with increased thickness
                if result['label'] == 'incorrect':
                    # Draw double rectangle for better visibility
                    cv2.rectangle(vis_image,
                                  (region['box']['x1'], region['box']['y1']),
                                  (region['box']['x2'], region['box']['y2']),
                                  color,
                                  4)  # Increased thickness

                    # Draw connecting line from box to feedback
                    feedback_pos = region['feedback_pos']
                    box_center_x = (region['box']['x1'] + region['box']['x2']) // 2
                    box_center_y = (region['box']['y1'] + region['box']['y2']) // 2

                    # Draw line with background for better visibility
                    cv2.line(vis_image,
                             (box_center_x, box_center_y),
                             (feedback_pos['x'], feedback_pos['y']),
                             (255, 255, 255),
                             3)  # White background line
                    cv2.line(vis_image,
                             (box_center_x, box_center_y),
                             (feedback_pos['x'], feedback_pos['y']),
                             color,
                             2)  # Colored line

                # Prepare feedback text
                status_text = f"{result['stichwort']}: {result['label'].upper()}"
                conf_text = f"Conf: {result['confidence']:.2f}"

                # Create more opaque background for text
                feedback_pos = region['feedback_pos']
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]

                # Draw background rectangle
                bg_pts = np.array([
                    [feedback_pos['x'] - 10, feedback_pos['y'] - text_size[1] - 15],
                    [feedback_pos['x'] + text_size[0] + 10, feedback_pos['y'] - text_size[1] - 15],
                    [feedback_pos['x'] + text_size[0] + 10, feedback_pos['y'] + 25],
                    [feedback_pos['x'] - 10, feedback_pos['y'] + 25]
                ], np.int32)

                overlay = vis_image.copy()
                cv2.fillPoly(overlay, [bg_pts], (255, 255, 255))
                cv2.addWeighted(overlay, 0.85, vis_image, 0.15, 0, vis_image)

                # Draw text with increased size and thickness
                cv2.putText(vis_image,
                            status_text,
                            (feedback_pos['x'], feedback_pos['y']),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9,  # Increased font size
                            color,
                            2)  # Increased thickness

                # Draw confidence with increased size
                cv2.putText(vis_image,
                            conf_text,
                            (feedback_pos['x'], feedback_pos['y'] + 20),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,  # Increased font size
                            color,
                            2)  # Increased thickness

            # Add timestamp at bottom with increased size
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(vis_image,
                        f"Analysis Time: {timestamp}",
                        (10, height - 20),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.8,  # Increased font size
                        (0, 0, 0),
                        2)  # Increased thickness

            # Save visualization with high quality
            cv2.imwrite(output_path, vis_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            self.logger.info(f"Saved visualization to {output_path}")

        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            raise

def main():
    """Example usage of inference"""
    try:
        # Initialize inference
        model_path = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/Multifusion/checkpoints04/best_model.pt"
        inference = SingleImageInference(model_path)

        # Path to test image
        image_path = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/data/01corrected/29.jpg"

        # Run inference
        results = inference.predict(image_path)

        # Print results
        print("\nInference Results:")
        print("-" * 50)
        for criterion, result in results['predictions'].items():
            print(f"\n{criterion}:")
            print(f"Lernziel: {result['lernziel']}")
            print(f"Stichwort: {result['stichwort']}")
            print(f"Kriterium: {result['kriterium']}")
            print(f"Label: {result['label']}")
            print(f"Confidence: {result['confidence']:.2f}")

        # Create visualization
        output_path = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/Multifusion/outputresult/inference_visualization10.jpg"
        inference.visualize_results(image_path, results, output_path)

    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()