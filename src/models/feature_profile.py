#!/usr/bin/env python3
"""
Feature Profile Base Classes for Novel Taxon Prediction

Provides abstract base classes and utilities for encoding feature profiles
into embeddings compatible with different KG embedding models.
"""

import torch
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEncoder(ABC):
    """
    Abstract base class for feature encoders

    Each model type (RotatE, ULTRA, RelGT) implements this interface
    to convert feature profiles into embeddings compatible with their architecture.
    """

    def __init__(self, model, vocabularies: Dict, device: str = 'cuda'):
        """
        Initialize feature encoder

        Args:
            model: Trained model instance
            vocabularies: Dictionary with entity2id, relation2id mappings
            device: Device to run on (cuda/cpu)
        """
        self.model = model
        self.vocabularies = vocabularies
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Extract vocabularies
        if 'entity2id' in vocabularies:
            self.entity2id = vocabularies['entity2id']
            self.id2entity = vocabularies.get('id2entity', {})
        elif 'entities' in vocabularies:
            self.entity2id = vocabularies['entities']
            self.id2entity = {v: k for k, v in self.entity2id.items()}
        else:
            raise ValueError("Vocabularies must contain 'entity2id' or 'entities'")

        if 'relation2id' in vocabularies:
            self.relation2id = vocabularies['relation2id']
            self.id2relation = vocabularies.get('id2relation', {})
        elif 'relations' in vocabularies:
            self.relation2id = vocabularies['relations']
            self.id2relation = {v: k for k, v in self.relation2id.items()}
        else:
            raise ValueError("Vocabularies must contain 'relation2id' or 'relations'")

        # Trait relations
        self.trait_relations = {'biolink:has_phenotype', 'biolink:capable_of'}

        logger.info(f"Initialized {self.__class__.__name__} with {len(self.entity2id)} entities")

    @abstractmethod
    def encode_features(self, features_dict: Dict[str, str]) -> torch.Tensor:
        """
        Encode feature profile into embedding

        Args:
            features_dict: Dictionary of features {trait_type: trait_value}

        Returns:
            Tensor embedding representing the feature profile
        """
        pass

    @abstractmethod
    def predict_media(
        self,
        feature_embedding: torch.Tensor,
        topk: int = 20,
        target_relation: str = 'biolink:occurs_in'
    ) -> List[Dict[str, Any]]:
        """
        Predict media from feature embedding

        Args:
            feature_embedding: Encoded feature tensor
            topk: Number of top predictions to return
            target_relation: Relation to predict (default: biolink:occurs_in)

        Returns:
            List of prediction dictionaries with scores
        """
        pass

    def get_feature_node_ids(self, features_dict: Dict[str, str]) -> List[int]:
        """
        Convert feature dictionary to entity IDs

        Args:
            features_dict: Dictionary of features {trait_type: trait_value}

        Returns:
            List of valid entity IDs
        """
        feature_ids = []

        for feat_type, feat_value in features_dict.items():
            feature_node = f"{feat_type}:{feat_value}"

            if feature_node in self.entity2id:
                feature_ids.append(self.entity2id[feature_node])
            else:
                logger.warning(f"Feature not in KG: {feature_node}")

        return feature_ids

    def get_media_entity_ids(self) -> List[int]:
        """
        Get all media entity IDs from vocabularies

        Returns:
            List of media entity IDs
        """
        media_ids = []
        media_prefix = 'medium:'

        for entity, ent_id in self.entity2id.items():
            if entity.startswith(media_prefix):
                media_ids.append(ent_id)

        logger.info(f"Found {len(media_ids)} media entities")
        return media_ids


class MeanPoolingEncoder(FeatureEncoder):
    """
    Simple feature encoder using mean pooling of entity embeddings

    This is a model-agnostic baseline that works with any model
    that has entity embeddings.
    """

    def encode_features(self, features_dict: Dict[str, str]) -> torch.Tensor:
        """
        Encode features via mean pooling of entity embeddings

        Args:
            features_dict: Dictionary of features

        Returns:
            Mean-pooled embedding tensor
        """
        feature_ids = self.get_feature_node_ids(features_dict)

        if not feature_ids:
            raise ValueError("No valid features found in KG")

        # Get entity embeddings
        with torch.no_grad():
            feature_ids_tensor = torch.tensor(feature_ids, device=self.device, dtype=torch.long)

            # Access entity embeddings from model
            if hasattr(self.model, 'entity_embedding'):
                embeddings = self.model.entity_embedding(feature_ids_tensor)
            else:
                raise AttributeError(f"Model {type(self.model)} does not have entity_embedding")

            # Mean pooling
            pooled_embedding = embeddings.mean(dim=0)

        logger.info(f"Encoded {len(feature_ids)} features -> embedding shape {pooled_embedding.shape}")
        return pooled_embedding

    def predict_media(
        self,
        feature_embedding: torch.Tensor,
        topk: int = 20,
        target_relation: str = 'biolink:occurs_in'
    ) -> List[Dict[str, Any]]:
        """
        Predict media using cosine similarity

        Args:
            feature_embedding: Encoded feature tensor
            topk: Number of top predictions
            target_relation: Target relation (not used in this baseline)

        Returns:
            List of predictions
        """
        media_ids = self.get_media_entity_ids()

        if not media_ids:
            raise ValueError("No media entities found in KG")

        with torch.no_grad():
            media_ids_tensor = torch.tensor(media_ids, device=self.device, dtype=torch.long)

            # Get media embeddings
            if hasattr(self.model, 'entity_embedding'):
                media_embeddings = self.model.entity_embedding(media_ids_tensor)
            else:
                raise AttributeError("Model does not have entity_embedding")

            # Compute cosine similarity
            feature_norm = feature_embedding / (feature_embedding.norm() + 1e-10)
            media_norms = media_embeddings / (media_embeddings.norm(dim=1, keepdim=True) + 1e-10)

            similarities = torch.matmul(media_norms, feature_norm)

            # Get top-k
            topk_actual = min(topk, len(media_ids))
            top_scores, top_indices = torch.topk(similarities, topk_actual)

        # Format predictions
        predictions = []
        for rank, (idx, score) in enumerate(zip(top_indices.cpu().numpy(), top_scores.cpu().numpy())):
            media_id = media_ids[idx]
            media_name = self.id2entity.get(media_id, f"entity_{media_id}")

            predictions.append({
                'subject': 'virtual_taxon',
                'predicate': target_relation,
                'object': media_name,
                'raw_score': float(score),
                'rank': rank + 1,
                'method': 'mean_pooling_cosine'
            })

        return predictions


def compute_softmax_and_logit(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add softmax and logit scores to predictions

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Updated predictions with softmax and logit scores
    """
    if not predictions:
        return predictions

    # Extract raw scores
    raw_scores = np.array([p['raw_score'] for p in predictions])

    # Compute softmax
    exp_scores = np.exp(raw_scores - np.max(raw_scores))  # Numerical stability
    softmax_scores = exp_scores / np.sum(exp_scores)

    # Compute logit (sigmoid)
    logit_scores = 1.0 / (1.0 + np.exp(-raw_scores))

    # Update predictions
    for i, pred in enumerate(predictions):
        pred['softmax'] = float(softmax_scores[i])
        pred['logit'] = float(logit_scores[i])

    return predictions


def add_confidence_annotations(
    predictions: List[Dict[str, Any]],
    feature_coverage: float,
    feature_count: int,
    warnings: List[str]
) -> List[Dict[str, Any]]:
    """
    Add confidence levels and warnings to predictions

    Args:
        predictions: List of predictions
        feature_coverage: Fraction of valid features
        feature_count: Number of features
        warnings: List of warning messages

    Returns:
        Updated predictions with confidence annotations
    """
    # Determine overall confidence level
    if feature_coverage >= 0.8 and feature_count >= 5 and not warnings:
        confidence = 'high'
    elif feature_coverage >= 0.6 and feature_count >= 3:
        confidence = 'medium'
    else:
        confidence = 'low'

    # Add to all predictions
    for pred in predictions:
        pred['confidence'] = confidence

        # Add warnings to top prediction only
        if pred['rank'] == 1 and warnings:
            pred['warnings'] = '; '.join(warnings)

    return predictions
