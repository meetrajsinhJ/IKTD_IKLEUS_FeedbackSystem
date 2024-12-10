import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
import torch

# Import from your existing modules
from LVmodule.Multifusion.imageProcessing import TechnicalDrawingProcessor, ProcessedDrawing
from LVmodule.Multifusion.textProcessing import AnnotationTextProcessor, FeedbackItem
from LVmodule.Multifusion.knowledgeEmbedding import TechnicalDrawingCriteriaEmbedder


@dataclass
class ProcessedSample:
    """Data class for a processed sample ready for model training"""
    sample_id: str
    incorrect_image_features: Dict
    correct_image_features: Dict
    annotation_features: List[Dict]
    knowledge_embeddings: Dict
    labels: Dict  # Ground truth information
    metadata: Dict


class TechnicalDrawingDataPipeline:
    """Pipeline for processing technical drawing data for ML model"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Initialize processors
        self.image_processor = TechnicalDrawingProcessor()
        self.text_processor = AnnotationTextProcessor()
        self.knowledge_embedder = TechnicalDrawingCriteriaEmbedder()

        # Load review criteria
        self.review_criteria = self._load_review_criteria()

    def _load_review_criteria(self) -> List[Dict]:
        """Load review criteria from file"""
        criteria = [
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
        return criteria

    def process_sample(self,
                       sample_id: str,
                       incorrect_path: str,
                       correct_path: str,
                       annotation_path: str) -> ProcessedSample:
        """Process a single sample (pair of drawings with annotations)"""
        try:
            # 1. Process images
            incorrect_processed, correct_processed = self.image_processor.process_drawing_pair(
                incorrect_path,
                correct_path,
                annotation_path
            )

            # 2. Process annotations
            annotation_data = self.text_processor.load_annotation_file(annotation_path)
            feedback_items = self.text_processor.process_annotations(annotation_data)

            # 3. Create knowledge embeddings
            embedded_criteria = self.knowledge_embedder.create_embeddings(
                self.knowledge_embedder.parse_review_table(self.review_criteria)
            )

            # 4. Extract features and combine
            processed_sample = self._combine_features(
                sample_id,
                incorrect_processed,
                correct_processed,
                feedback_items,
                embedded_criteria
            )

            return processed_sample

        except Exception as e:
            self.logger.error(f"Error processing sample {sample_id}: {str(e)}")
            raise

    def _combine_features(self,
                          sample_id: str,
                          incorrect_processed: ProcessedDrawing,
                          correct_processed: ProcessedDrawing,
                          feedback_items: List[FeedbackItem],
                          embedded_criteria: Dict) -> ProcessedSample:
        """Combine features from different sources"""

        # Extract image features
        incorrect_features = {
            'visual_features': incorrect_processed.features,
            'regions': self._process_regions(incorrect_processed.annotation_regions),
            'structural_features': {
                'lines': len(incorrect_processed.detected_lines),
                'contours': len(incorrect_processed.detected_contours)
            }
        }

        correct_features = {
            'visual_features': correct_processed.features,
            'structural_features': {
                'lines': len(correct_processed.detected_lines),
                'contours': len(correct_processed.detected_contours)
            }
        }

        # Process annotations
        annotation_features = []
        for item in feedback_items:
            feature = {
                'criterion_id': item.criterion_id,
                'measurements': item.measurements,
                'location': item.bbox,
                'is_verified': item.is_verified,
                'embedding': embedded_criteria.get(item.criterion_id, {}).get('embedding', None)
            }
            annotation_features.append(feature)

        # Create labels
        labels = self._create_labels(feedback_items, incorrect_processed.annotations)

        return ProcessedSample(
            sample_id=sample_id,
            incorrect_image_features=incorrect_features,
            correct_image_features=correct_features,
            annotation_features=annotation_features,
            knowledge_embeddings=embedded_criteria,
            labels=labels,
            metadata={
                'timestamp': None,  # Add if needed
                'source': 'technical_drawing_pipeline'
            }
        )

    def _process_regions(self, regions: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """Process annotation regions to extract region-specific features"""
        processed_regions = {}

        for region_id, region_img in regions.items():
            features = {
                'mean_intensity': np.mean(region_img),
                'std_intensity': np.std(region_img),
                'size': region_img.shape,
                'edge_density': np.mean(cv2.Canny(region_img, 50, 150)) / 255.0
            }
            processed_regions[region_id] = features

        return processed_regions

    def _create_labels(self,
                       feedback_items: List[FeedbackItem],
                       annotations: List) -> Dict:
        """Create labels from feedback items and annotations"""
        labels = {
            'criteria_scores': defaultdict(float),
            'region_labels': defaultdict(dict),
            'verification_status': defaultdict(bool)
        }

        for item in feedback_items:
            criterion_id = item.criterion_id
            labels['verification_status'][criterion_id] = item.is_verified

            # Add region-specific labels if available
            if item.bbox:
                labels['region_labels'][criterion_id] = {
                    'bbox': item.bbox,
                    'measurements': item.measurements
                }

        return dict(labels)

    def process_dataset(self, dataset_config: Dict) -> List[ProcessedSample]:
        """Process entire dataset based on configuration"""
        processed_samples = []

        try:
            for sample in dataset_config['samples']:
                sample_id = sample['id']
                self.logger.info(f"Processing sample {sample_id}")

                processed_sample = self.process_sample(
                    sample_id,
                    sample['incorrect_path'],
                    sample['correct_path'],
                    sample['annotation_path']
                )

                processed_samples.append(processed_sample)

            self.logger.info(f"Processed {len(processed_samples)} samples")
            return processed_samples

        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")
            raise

    def save_processed_data(self,
                            processed_samples: List[ProcessedSample],
                            output_dir: Path):
        """Save processed data to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in processed_samples:
            sample_dir = output_dir / sample.sample_id
            sample_dir.mkdir(exist_ok=True)

            # Save features
            np.save(sample_dir / 'incorrect_features.npy',
                    self._convert_to_numpy(sample.incorrect_image_features))
            np.save(sample_dir / 'correct_features.npy',
                    self._convert_to_numpy(sample.correct_image_features))
            np.save(sample_dir / 'annotation_features.npy',
                    self._convert_to_numpy(sample.annotation_features))

            # Save labels
            np.save(sample_dir / 'labels.npy',
                    self._convert_to_numpy(sample.labels))

            # Save metadata
            with open(sample_dir / 'metadata.json', 'w') as f:
                json.dump(sample.metadata, f)

    def _convert_to_numpy(self, data):
        """Convert various data types to numpy arrays for saving"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        elif isinstance(data, dict):
            return {k: self._convert_to_numpy(v) for k, v in data.items()}
        return data


def main():
    """Example usage of the pipeline"""
    # Setup paths
    base_path = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01")
    output_dir = Path("processed_data")

    # Dataset configuration
    dataset_config = {
        'samples': [
            {
                'id': '1',
                'incorrect_path': str(base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/1.jpg"),
                'correct_path': str(base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/1.jpg"),
                'annotation_path': str(base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/1_annotations.json")
            },
            {
                'id': '2',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/2.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/2.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/3_annotations.json")
            },
            {
                'id': '3',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/3.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/3.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/3_annotations.json")
            },
            {
                'id': '4',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/4.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/4.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/4_annotations.json")
            },
            {
                'id': '5',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/5.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/5.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/5_annotations.json")
            },
            {
                'id': '6',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/6.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/6.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/6_annotations.json")
            },
            {
                'id': '7',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/7.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/7.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/7_annotations.json")
            },
            {
                'id': '8',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/8.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/8.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/8_annotations.json")
            },
            {
                'id': '9',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/9.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/9.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/9_annotations.json")
            },
            {
                'id': '10',
                'incorrect_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawuncorrected/10.jpg"),
                'correct_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/01Schraubedrawcorrected/10.jpg"),
                'annotation_path': str(
                    base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/a01/mapped_output/annotations/10_annotations.json")
            },

        ]
    }

    try:
        # Initialize pipeline
        pipeline = TechnicalDrawingDataPipeline(base_path)

        # Process dataset
        processed_samples = pipeline.process_dataset(dataset_config)

        # Save processed data
        pipeline.save_processed_data(processed_samples, output_dir)

        print(f"Successfully processed {len(processed_samples)} samples")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()