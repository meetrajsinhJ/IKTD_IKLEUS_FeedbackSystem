import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
import torch
import glob
import os

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

    def auto_generate_dataset_config(self) -> Dict:
        """Automatically generate dataset configuration for all samples"""
        dataset_config = {'samples': []}

        try:
            # Find all subdirectories in the data path that match the pattern a01, a02, etc.
            assignment_dirs = sorted(glob.glob(str(self.base_path / "a*")))

            for assignment_dir in assignment_dirs:
                assignment_path = Path(assignment_dir)

                # Find all uncorrected drawings
                uncorrected_dir = assignment_path / "01uncorrected"
                corrected_dir = assignment_path / "01corrected"
                annotations_dir = assignment_path / "annotationsBB"

                if not all([uncorrected_dir.exists(), corrected_dir.exists(), annotations_dir.exists()]):
                    self.logger.warning(f"Skipping {assignment_dir} - missing required directories")
                    continue

                # Get all image files
                image_files = sorted([f for f in uncorrected_dir.glob("*.jpg")])

                for image_file in image_files:
                    sample_id = image_file.stem  # Get filename without extension

                    # Construct paths for this sample
                    incorrect_path = uncorrected_dir / f"{sample_id}.jpg"
                    correct_path = corrected_dir / f"{sample_id}.jpg"
                    annotation_path = annotations_dir / f"{sample_id}_annotations.json"

                    # Verify all files exist
                    if not all([incorrect_path.exists(), correct_path.exists(), annotation_path.exists()]):
                        self.logger.warning(f"Skipping sample {sample_id} - missing required files")
                        continue

                    # Add to dataset configuration
                    dataset_config['samples'].append({
                        'id': sample_id,
                        'incorrect_path': str(incorrect_path),
                        'correct_path': str(correct_path),
                        'annotation_path': str(annotation_path)
                    })

                    self.logger.info(f"Added sample {sample_id} to dataset configuration")

        except Exception as e:
            self.logger.error(f"Error generating dataset configuration: {str(e)}")
            raise

        self.logger.info(f"Generated configuration for {len(dataset_config['samples'])} samples")
        return dataset_config

    def process_sample(self,
                       sample_id: str,
                       incorrect_path: str,
                       correct_path: str,
                       annotation_path: str) -> 'ProcessedSample':
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
            criteria = self.knowledge_embedder.parse_review_table(self.review_criteria)
            embedded_criteria = self.knowledge_embedder.create_embeddings(criteria)

            # 4. Extract features and combine
            return self._combine_features(
                sample_id,
                incorrect_processed,
                correct_processed,
                feedback_items,
                embedded_criteria
            )

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
            criterion_id = item.criterion_id
            embedded_criterion = embedded_criteria.get(criterion_id)

            if embedded_criterion:
                feature = {
                    'criterion_id': criterion_id,
                    'measurements': item.measurements,
                    'is_verified': item.is_verified,
                    'embedding': embedded_criterion.embedding.tolist(),
                    'technical_category': embedded_criterion.technical_category,
                    'complexity_score': embedded_criterion.complexity_score
                }
                annotation_features.append(feature)

        # Create labels
        labels = {
            'verification_status': {item.criterion_id: item.is_verified for item in feedback_items}
        }

        return ProcessedSample(
            sample_id=sample_id,
            incorrect_image_features=incorrect_features,
            correct_image_features=correct_features,
            annotation_features=annotation_features,
            knowledge_embeddings=embedded_criteria,
            labels=labels,
            metadata={
                'source': 'technical_drawing_pipeline',
                'processed_annotations': len(annotation_features),
                'processed_criteria': len(embedded_criteria)
            }
        )

    def process_dataset(self, output_dir: Path, batch_size: int = 10) -> None:
        """Process entire dataset with batching"""
        try:
            # Get dataset configuration
            dataset_config = self.auto_generate_dataset_config()
            total_samples = len(dataset_config['samples'])

            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process in batches
            for i in range(0, total_samples, batch_size):
                batch_samples = dataset_config['samples'][i:i + batch_size]
                self.logger.info(
                    f"Processing batch {i // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}")

                for sample in batch_samples:
                    try:
                        processed_sample = self.process_sample(
                            sample['id'],
                            sample['incorrect_path'],
                            sample['correct_path'],
                            sample['annotation_path']
                        )

                        # Save processed sample
                        sample_dir = output_dir / sample['id']
                        sample_dir.mkdir(exist_ok=True)

                        # Save features
                        np.save(sample_dir / 'incorrect_features.npy',
                                self._convert_to_numpy(processed_sample.incorrect_image_features))
                        np.save(sample_dir / 'correct_features.npy',
                                self._convert_to_numpy(processed_sample.correct_image_features))
                        np.save(sample_dir / 'annotation_features.npy',
                                self._convert_to_numpy(processed_sample.annotation_features))
                        np.save(sample_dir / 'labels.npy',
                                self._convert_to_numpy(processed_sample.labels))

                        # Save metadata
                        with open(sample_dir / 'metadata.json', 'w') as f:
                            json.dump(processed_sample.metadata, f)

                        self.logger.info(f"Successfully processed and saved sample {sample['id']}")

                    except Exception as e:
                        self.logger.error(f"Error processing sample {sample['id']}: {str(e)}")
                        continue

                self.logger.info(f"Completed processing batch {i // batch_size + 1}")

        except Exception as e:
            self.logger.error(f"Error in dataset processing: {str(e)}")
            raise

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
    """Process the complete dataset"""
    try:
        # Setup paths
        base_path = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/data")
        output_dir = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/processed_data")

        # Initialize pipeline
        pipeline = TechnicalDrawingDataPipeline(base_path)

        # Process dataset with batching
        pipeline.process_dataset(
            output_dir=output_dir,
            batch_size=10  # Adjust based on your system's memory
        )

        logging.info("Dataset processing completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()