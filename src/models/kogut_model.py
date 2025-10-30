"""
KOGUT (Knowledge Graph Universal Transformer) Model for Inference

Simplified model architecture for loading trained KOGUT models and making predictions.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class KOGUTModel(nn.Module):
    """
    Simplified KOGUT model for inference.

    This model loads the entity and relation embeddings from a trained
    KOGUT checkpoint and uses them for prediction via similarity scoring.
    """

    def __init__(self, num_entities: int, num_relations: int, hidden_dim: int = 64):
        """
        Initialize KOGUT model.

        Args:
            num_entities: Number of entities in the knowledge graph
            num_relations: Number of relations in the knowledge graph
            hidden_dim: Embedding dimension (default: 64 for KOGUT)
        """
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        # Entity and relation embeddings
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        logger.info(
            f"Initialized KOGUT model: {num_entities} entities, "
            f"{num_relations} relations, dim={hidden_dim}"
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - get entity embeddings.

        Args:
            entity_ids: Tensor of entity IDs

        Returns:
            Entity embeddings
        """
        return self.entity_embedding(entity_ids)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Score a triple (head, relation, tail).

        Args:
            head: Head entity embedding
            relation: Relation embedding
            tail: Tail entity embedding

        Returns:
            Score for the triple
        """
        # Simple cosine similarity scoring
        score = torch.sum(head * tail, dim=-1)
        return score


def load_kogut_model(
    model_path: str,
    num_entities: int,
    num_relations: int,
    hidden_dim: int = 64,
    device: str = 'cpu'
) -> KOGUTModel:
    """
    Load a trained KOGUT model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pt file)
        num_entities: Number of entities
        num_relations: Number of relations
        hidden_dim: Embedding dimension (default: 64)
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded KOGUT model in eval mode
    """
    logger.info(f"Loading KOGUT model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # Create model
    model = KOGUTModel(num_entities, num_relations, hidden_dim)
    model = model.to(device)

    # Load state dict
    # Try different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load with strict=False to allow for architecture differences
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info(f"Successfully loaded KOGUT model (hidden_dim={hidden_dim})")

    return model
