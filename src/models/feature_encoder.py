"""
Feature encoder for converting trait profiles to predictions.

Based on MicroGrowLink's feature encoding approach.
"""

import torch
import numpy as np
import logging
import json
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class KOGUTFeatureEncoder:
    """
    Encodes microbial trait features and predicts growth media using KOGUT model.
    """

    def __init__(self, model, vocabularies: Dict, device: str = 'cpu'):
        """
        Initialize feature encoder.

        Args:
            model: Trained KOGUT model
            vocabularies: Dictionary with entities and relations mappings
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Extract vocabularies
        if 'entities' in vocabularies:
            self.entity2id = vocabularies['entities']
        elif 'entity2id' in vocabularies:
            self.entity2id = vocabularies['entity2id']
        else:
            raise ValueError("Vocabularies must contain 'entities' or 'entity2id'")

        if 'relations' in vocabularies:
            self.relation2id = vocabularies['relations']
        elif 'relation2id' in vocabularies:
            self.relation2id = vocabularies['relation2id']
        else:
            raise ValueError("Vocabularies must contain 'relations' or 'relation2id'")

        # Create reverse mappings
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        logger.info(
            f"Initialized KOGUT encoder with {len(self.entity2id)} entities, "
            f"{len(self.relation2id)} relations"
        )

    def encode_features(self, features_dict: Dict[str, str]) -> torch.Tensor:
        """
        Encode trait features into an embedding.

        Args:
            features_dict: Dictionary of {feature_type: feature_value}
                         e.g., {'temperature': 'mesophilic', 'oxygen': 'aerobe'}

        Returns:
            Aggregated feature embedding tensor
        """
        feature_ids = []

        # Convert features to entity IDs
        for feat_type, feat_value in features_dict.items():
            feature_node = f"{feat_type}:{feat_value}"

            if feature_node in self.entity2id:
                feature_ids.append(self.entity2id[feature_node])
            else:
                logger.warning(f"Feature not in knowledge graph: {feature_node}")

        if not feature_ids:
            raise ValueError("No valid features found in knowledge graph")

        logger.info(f"Encoded {len(feature_ids)} features")

        # Get embeddings for feature nodes
        with torch.no_grad():
            feature_ids_tensor = torch.tensor(
                feature_ids,
                device=self.device,
                dtype=torch.long
            )

            # Get embeddings from model
            feature_embeddings = self.model.entity_embedding(feature_ids_tensor)

            # Aggregate using mean pooling
            aggregated = feature_embeddings.mean(dim=0)

        return aggregated

    def predict_media(
        self,
        feature_embedding: torch.Tensor,
        topk: int = 20
    ) -> List[Dict]:
        """
        Predict growth media from feature embedding.

        Args:
            feature_embedding: Aggregated feature embedding
            topk: Number of top predictions to return

        Returns:
            List of predictions with scores
        """
        # Get all media entity IDs
        media_ids = []
        media_prefix = 'medium:'

        for entity, ent_id in self.entity2id.items():
            if entity.startswith(media_prefix):
                media_ids.append(ent_id)

        if not media_ids:
            raise ValueError("No media entities found in knowledge graph")

        logger.info(f"Scoring {len(media_ids)} media entities")

        with torch.no_grad():
            media_ids_tensor = torch.tensor(
                media_ids,
                device=self.device,
                dtype=torch.long
            )

            # Get media embeddings
            media_embeddings = self.model.entity_embedding(media_ids_tensor)

            # Compute similarity scores (cosine similarity)
            feature_norm = feature_embedding / (feature_embedding.norm() + 1e-10)
            media_norms = media_embeddings / (media_embeddings.norm(dim=1, keepdim=True) + 1e-10)

            scores = torch.matmul(media_norms, feature_norm)

            # Get top-k
            topk_actual = min(topk, len(media_ids))
            top_scores, top_indices = torch.topk(scores, topk_actual)

        # Format predictions
        predictions = []
        for rank, (idx, score) in enumerate(
            zip(top_indices.cpu().numpy(), top_scores.cpu().numpy())
        ):
            media_id = media_ids[idx]
            media_name = self.id2entity.get(media_id, f"entity_{media_id}")

            predictions.append({
                'subject': 'virtual_taxon',
                'predicate': 'biolink:occurs_in',
                'object': media_name,
                'raw_score': float(score),
                'rank': rank + 1
            })

        return predictions


def compute_softmax_and_logit(predictions: List[Dict]) -> List[Dict]:
    """
    Add softmax and logit (sigmoid) scores to predictions.

    Args:
        predictions: List of prediction dictionaries with raw_score

    Returns:
        Updated predictions with softmax and logit scores
    """
    if not predictions:
        return predictions

    # Extract raw scores
    raw_scores = np.array([p['raw_score'] for p in predictions])

    # Compute softmax (normalized probabilities)
    exp_scores = np.exp(raw_scores - np.max(raw_scores))  # Numerical stability
    softmax_scores = exp_scores / np.sum(exp_scores)

    # Compute logit (sigmoid) - confidence score
    logit_scores = 1.0 / (1.0 + np.exp(-raw_scores))

    # Update predictions
    for i, pred in enumerate(predictions):
        pred['softmax'] = float(softmax_scores[i])
        pred['logit'] = float(logit_scores[i])

    return predictions


def add_confidence_annotations(
    predictions: List[Dict],
    feature_count: int,
    valid_feature_count: int
) -> List[Dict]:
    """
    Add confidence levels to predictions based on feature quality.

    Args:
        predictions: List of predictions
        feature_count: Total number of input features
        valid_feature_count: Number of features found in KG

    Returns:
        Updated predictions with confidence levels
    """
    if not predictions:
        return predictions

    # Compute feature coverage
    coverage = valid_feature_count / feature_count if feature_count > 0 else 0

    # Determine confidence level
    if coverage >= 0.8 and valid_feature_count >= 5:
        confidence = 'high'
    elif coverage >= 0.6 and valid_feature_count >= 3:
        confidence = 'medium'
    else:
        confidence = 'low'

    # Add confidence to all predictions
    for pred in predictions:
        pred['confidence'] = confidence

    return predictions


def load_vocabularies(vocab_path: Path) -> Dict:
    """
    Load KOGUT vocabularies from JSON file.

    Args:
        vocab_path: Path to vocabularies.json

    Returns:
        Dictionary with entities and relations mappings
    """
    logger.info(f"Loading vocabularies from {vocab_path}")

    with open(vocab_path, 'r') as f:
        vocabularies = json.load(f)

    # Standardize format
    if 'entity2id' in vocabularies and 'entities' not in vocabularies:
        vocabularies['entities'] = vocabularies['entity2id']

    if 'relation2id' in vocabularies and 'relations' not in vocabularies:
        vocabularies['relations'] = vocabularies['relation2id']

    num_entities = len(vocabularies.get('entities', vocabularies.get('entity2id', {})))
    num_relations = len(vocabularies.get('relations', vocabularies.get('relation2id', {})))

    logger.info(f"Loaded {num_entities:,} entities and {num_relations} relations")

    return vocabularies
