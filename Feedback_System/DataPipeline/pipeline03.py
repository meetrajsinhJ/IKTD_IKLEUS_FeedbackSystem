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


    def _process_regions(self, regions: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """Process annotation regions to extract region-specific features"""
        processed_regions = {}

        try:
            for region_id, region_img in regions.items():
                if region_img is None or region_img.size == 0:
                    self.logger.warning(f"Empty or invalid region for ID {region_id}")
                    continue

                try:
                    # Convert to grayscale if needed
                    if len(region_img.shape) == 3:
                        gray_region = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_region = region_img

                    # Basic statistical features
                    basic_features = {
                        'mean_intensity': float(np.mean(gray_region)),
                        'std_intensity': float(np.std(gray_region)),
                        'min_intensity': float(np.min(gray_region)),
                        'max_intensity': float(np.max(gray_region)),
                        'size': list(region_img.shape),
                    }

                    # Edge features
                    edges = cv2.Canny(gray_region, 50, 150)
                    edge_features = {
                        'edge_density': float(np.mean(edges) / 255.0),
                        'edge_std': float(np.std(edges) / 255.0)
                    }

                    # Texture features using GLCM
                    texture_features = self._compute_texture_features(gray_region)

                    # Combine all features
                    processed_regions[region_id] = {
                        **basic_features,
                        **edge_features,
                        **texture_features
                    }

                except Exception as e:
                    self.logger.error(f"Error processing region {region_id}: {str(e)}")
                    processed_regions[region_id] = {
                        'error': str(e),
                        'size': list(region_img.shape) if region_img is not None else None
                    }

            return processed_regions

        except Exception as e:
            self.logger.error(f"Error in region processing: {str(e)}")
            return {}

    def _compute_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Compute texture features for a grayscale image region"""
        try:
            # Normalize image
            normalized = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

            # Calculate histogram
            hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)  # Avoid division by zero

            # Compute basic texture metrics
            features = {
                'texture_entropy': float(-np.sum(hist * np.log2(hist + 1e-7))),
                'texture_energy': float(np.sum(hist ** 2)),
                'texture_contrast': float(np.sum((np.arange(256) - np.mean(normalized)) ** 2 * hist))
            }

            return features

        except Exception as e:
            self.logger.error(f"Error computing texture features: {str(e)}")
            return {
                'texture_entropy': 0.0,
                'texture_energy': 0.0,
                'texture_contrast': 0.0
            }

    def process_sample(self,
                       sample_id: str,
                       incorrect_path: str,
                       correct_path: str,
                       annotation_path: str) -> ProcessedSample:
        """Process a single sample (pair of drawings with annotations)"""
        try:
            self.logger.info(f"Processing images for sample {sample_id}")
            # 1. Process images
            incorrect_processed, correct_processed = self.image_processor.process_drawing_pair(
                incorrect_path,
                correct_path,
                annotation_path
            )

            self.logger.info(f"Processing annotations for sample {sample_id}")
            # 2. Process annotations
            annotation_data = self.text_processor.load_annotation_file(annotation_path)
            feedback_items = self.text_processor.process_annotations(annotation_data)

            self.logger.info(f"Creating knowledge embeddings for sample {sample_id}")
            # 3. Create knowledge embeddings
            criteria = self.knowledge_embedder.parse_review_table(self.review_criteria)
            embedded_criteria = self.knowledge_embedder.create_embeddings(criteria)

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
            criterion_id = int(item.criterion_id)
            embedded_criterion = embedded_criteria.get(criterion_id)

            feature = {
                'criterion_id': criterion_id,
                'measurements': item.measurements,
                'location': item.bbox,
                'is_verified': item.is_verified,
            }

            # Add embedding features if available
            if embedded_criterion is not None:
                feature.update({
                    'embedding': embedded_criterion.embedding.tolist(),
                    'technical_category': embedded_criterion.technical_category,
                    'complexity_score': embedded_criterion.complexity_score
                })

            annotation_features.append(feature)

        # Create labels
        labels = self._create_labels(feedback_items, incorrect_processed.annotations)

        # Process knowledge embeddings into a suitable format
        knowledge_embeddings = {}
        for criterion_id, criterion in embedded_criteria.items():
            knowledge_embeddings[criterion_id] = {
                'embedding': criterion.embedding.tolist(),
                'technical_category': criterion.technical_category,
                'complexity_score': criterion.complexity_score,
                'related_criteria': [
                    {'id': rel_id, 'similarity': float(sim)}
                    for rel_id, sim in criterion.related_criteria
                ]
            }

        return ProcessedSample(
            sample_id=sample_id,
            incorrect_image_features=incorrect_features,
            correct_image_features=correct_features,
            annotation_features=annotation_features,
            knowledge_embeddings=knowledge_embeddings,
            labels=labels,
            metadata={
                'source': 'technical_drawing_pipeline',
                'processed_annotations': len(annotation_features),
                'processed_criteria': len(knowledge_embeddings)
            }
        )

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
    try:
        # Setup paths
        base_path = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/data")
        output_dir = Path("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/processed_data")

        # Dataset configuration
        dataset_config = {'samples': []}

        # Directory paths
        uncorrected_dir = base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/data/01uncorrected"
        corrected_dir = base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/data/01corrected"
        annotations_dir = base_path / "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/DataPrep/data/annotationsBB"

        # Get all uncorrected images (they will be our reference for finding pairs)
        image_files = sorted(uncorrected_dir.glob("*.jpg"))

        for image_path in image_files:
            image_id = image_path.stem  # Get filename without extension

            # Construct corresponding paths
            incorrect_path = uncorrected_dir / f"{image_id}.jpg"
            correct_path = corrected_dir / f"{image_id}.jpg"
            annotation_path = annotations_dir / f"{image_id}.json"

            # Verify all files exist
            if all([incorrect_path.exists(), correct_path.exists(), annotation_path.exists()]):
                dataset_config['samples'].append({
                    'id': image_id,
                    'incorrect_path': str(incorrect_path),
                    'correct_path': str(correct_path),
                    'annotation_path': str(annotation_path)
                })
                print(f"Added sample {image_id} to dataset configuration")
            else:
                print(f"Skipping sample {image_id} - missing required files")

        # Initialize pipeline
        pipeline = TechnicalDrawingDataPipeline(base_path)

        # Process dataset
        processed_samples = pipeline.process_dataset(dataset_config)

        # Save processed data
        pipeline.save_processed_data(processed_samples, output_dir)

        print(f"Successfully processed {len(dataset_config['samples'])} samples")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()