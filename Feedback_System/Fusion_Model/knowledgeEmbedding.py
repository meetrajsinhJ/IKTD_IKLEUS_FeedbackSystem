import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import defaultdict
import re

@dataclass
class TechnicalCriterion:
    """Structure for technical drawing review criteria"""
    nr: int
    lernziel: str  # Learning objective (e.g., "Schnitt-darstellung")
    stichwort: str  # Keyword (e.g., "Schraube allgemein")
    kriterium: str  # Detailed criterion
    measurements: List[Dict[str, float]]  # Extracted measurements
    features: List[str]  # Technical features mentioned
    related_standards: List[str]  # Related technical standards


@dataclass
class EmbeddedCriterion:
    """Structure for embedded technical criterion"""
    criterion: TechnicalCriterion
    embedding: np.ndarray
    related_criteria: List[Tuple[int, float]]
    technical_category: str
    complexity_score: float


class TechnicalDrawingCriteriaEmbedder:
    """Specialized embedder for technical drawing review criteria"""

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer(model_name)

        # Technical drawing specific categories
        self.technical_categories = {
            'geometric_features': [
                'durchmesser', 'länge', 'fase', 'winkel', 'abstand',
                'd1', 'd2', 'd3', 'e1', 'körperkante'
            ],
            'thread_features': [
                'gewinde', 'gewindekern', 'gewindereserve', 'schraube',
                'grundloch'
            ],
            'visualization': [
                'schraffur', 'schnitt', 'darstellung', 'strichlinie',
                'volllinie'
            ],
            'standards': [
                'norm', 'maß', 'toleranz', '±', 'p'
            ]
        }

        # Measurement patterns
        self.measurement_patterns = {
            'diameter': [
                r'd[1-3]\s*=\s*(\d+(?:,\d+)?)',
                r'durchmesser\s*=?\s*(\d+(?:,\d+)?)'
            ],
            'length': [
                r'l\s*=\s*(\d+(?:,\d+)?)',
                r'länge\s*=?\s*(\d+(?:,\d+)?)'
            ],
            'angle': [
                r'(\d+(?:,\d+)?)\s*°',
                r'winkel\s*=?\s*(\d+(?:,\d+)?)'
            ]
        }

    def parse_review_table(self, criteria_list: List[Dict]) -> List[TechnicalCriterion]:
        """Parse the technical drawing review criteria"""
        parsed_criteria = []

        for item in criteria_list:
            # Extract measurements and features from criterion text
            measurements = self._extract_measurements(item['kriterium'])
            features = self._extract_technical_features(item['kriterium'])
            standards = self._extract_standards(item['kriterium'])

            criterion = TechnicalCriterion(
                nr=int(item['nr']),
                lernziel=item['lernziel'],
                stichwort=item['stichwort'],
                kriterium=item['kriterium'],
                measurements=measurements,
                features=features,
                related_standards=standards
            )
            parsed_criteria.append(criterion)

        return parsed_criteria

    def _extract_measurements(self, text: str) -> List[Dict[str, float]]:
        """Extract measurement specifications from criterion text"""
        measurements = []
        text_lower = text.lower()

        for mtype, patterns in self.measurement_patterns.items():
            for pattern in patterns:
                import re
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    value = match.group(1).replace(',', '.')
                    try:
                        measurements.append({
                            'type': mtype,
                            'value': float(value),
                            'original_text': match.group(0)
                        })
                    except ValueError:
                        continue

        return measurements

    def _extract_technical_features(self, text: str) -> List[str]:
        """Extract technical features from criterion text"""
        features = []
        text_lower = text.lower()

        for category, terms in self.technical_categories.items():
            for term in terms:
                if term in text_lower:
                    features.append(term)

        return list(set(features))

    def _extract_standards(self, text: str) -> List[str]:
        """Extract referenced technical standards"""
        standards = []
        text_lower = text.lower()

        # Common technical drawing standards
        standard_patterns = [
            r'(din\s*\d+)',
            r'(iso\s*\d+)',
            r'(en\s*\d+)'
        ]

        for pattern in standard_patterns:
            matches = re.finditer(pattern, text_lower)
            standards.extend(match.group(1) for match in matches)

        return standards

    def create_embeddings(self, criteria: List[TechnicalCriterion]) -> Dict[int, EmbeddedCriterion]:
        """Create embeddings for technical criteria"""
        embedded_criteria = {}

        # Create text representations for embedding
        texts = [self._create_embedding_text(c) for c in criteria]

        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=True)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings.cpu().numpy())

        # Process each criterion
        for idx, criterion in enumerate(criteria):
            # Find related criteria
            related = self._find_related_criteria(
                idx, similarity_matrix, criteria
            )

            # Determine technical category
            category = self._determine_technical_category(criterion)

            # Calculate complexity score
            complexity = self._calculate_complexity_score(criterion)

            embedded_criteria[criterion.nr] = EmbeddedCriterion(
                criterion=criterion,
                embedding=embeddings[idx].cpu().numpy(),
                related_criteria=related,
                technical_category=category,
                complexity_score=complexity
            )

        return embedded_criteria

    def _create_embedding_text(self, criterion: TechnicalCriterion) -> str:
        """Create comprehensive text for embedding"""
        parts = [
            criterion.lernziel,
            criterion.stichwort,
            criterion.kriterium,
            ' '.join(criterion.features),
            ' '.join(f"{m['type']}_{m['value']}" for m in criterion.measurements)
        ]
        return ' '.join(parts)

    def _find_related_criteria(self,
                               idx: int,
                               similarity_matrix: np.ndarray,
                               criteria: List[TechnicalCriterion],
                               threshold: float = 0.5) -> List[Tuple[int, float]]:
        """Find related technical criteria"""
        related = []

        for i, similarity in enumerate(similarity_matrix[idx]):
            if i != idx and similarity >= threshold:
                related.append((criteria[i].nr, float(similarity)))

        return sorted(related, key=lambda x: x[1], reverse=True)

    def _determine_technical_category(self, criterion: TechnicalCriterion) -> str:
        """Determine primary technical category"""
        category_counts = defaultdict(int)

        for feature in criterion.features:
            for category, terms in self.technical_categories.items():
                if feature in terms:
                    category_counts[category] += 1

        if category_counts:
            return max(category_counts.items(), key=lambda x: x[1])[0]
        return "general"

    def _calculate_complexity_score(self, criterion: TechnicalCriterion) -> float:
        """Calculate technical complexity score"""
        score = 0.0

        # Add points for measurements
        score += len(criterion.measurements) * 0.2

        # Add points for technical features
        score += len(criterion.features) * 0.15

        # Add points for standards
        score += len(criterion.related_standards) * 0.25

        # Add points for specific technical terms
        technical_terms = ['toleranz', 'gewinde', 'phase', 'schnitt']
        text_lower = criterion.kriterium.lower()
        score += sum(0.1 for term in technical_terms if term in text_lower)

        return min(score, 1.0)  # Normalize to [0, 1]


def main():
    """Example usage with complete review table criteria"""

    # Complete review criteria from the table
    review_criteria = [
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

    embedder = TechnicalDrawingCriteriaEmbedder()

    try:
        # Parse criteria
        criteria = embedder.parse_review_table(review_criteria)
        print(f"\nParsed {len(criteria)} criteria")

        # Create embeddings
        embedded_criteria = embedder.create_embeddings(criteria)

        # Print analysis
        print("\nCriteria Analysis:")
        print("-" * 50)

        for nr, embedded in embedded_criteria.items():
            print(f"\nCriterion {nr}:")
            print(f"Category: {embedded.technical_category}")
            print(f"Complexity Score: {embedded.complexity_score:.2f}")
            print("Measurements:", embedded.criterion.measurements)
            print("Features:", embedded.criterion.features)
            print("Related Criteria:", [
                (crit_id, round(sim, 3))
                for crit_id, sim in embedded.related_criteria
            ])

        # Additional analysis
        print("\nCross-Reference Analysis:")
        print("-" * 50)

        # Group criteria by category
        category_groups = defaultdict(list)
        for nr, embedded in embedded_criteria.items():
            category_groups[embedded.technical_category].append(nr)

        print("\nCriteria by Category:")
        for category, criteria_ids in category_groups.items():
            print(f"\n{category}:")
            print(f"Criteria: {criteria_ids}")

        # Find highly related criteria pairs
        print("\nHighly Related Criteria Pairs (similarity > 0.7):")
        for nr, embedded in embedded_criteria.items():
            for related_id, similarity in embedded.related_criteria:
                if similarity > 0.7:
                    print(f"Criteria {nr} and {related_id}: {round(similarity, 3)}")

    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()