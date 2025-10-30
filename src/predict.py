"""
Prediction module for KOGUT model.

Loads KOGUT model directly and makes predictions without subprocess calls.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple
import config
import duckdb

from src.models.kogut_model import load_kogut_model
from src.models.feature_encoder import (
    KOGUTFeatureEncoder,
    load_vocabularies,
    compute_softmax_and_logit,
    add_confidence_annotations
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicroGrowPredictor:
    """
    Predictor for microbial growth media using KOGUT model.
    """

    def __init__(
        self,
        model_path: Path = None,
        data_path: Path = None,
        model_type: str = "kogut",
        device: str = "cpu"
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model .pt file
            data_path: Path to data directory
            model_type: Model type ('kogut')
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = model_path or config.MODEL_PATH
        self.data_path = data_path or config.DATA_PATH
        self.model_type = model_type
        self.device = device

        # Validate paths
        self._validate_paths()

        # Load vocabularies
        vocab_path = self.data_path / "kogut" / "vocabularies.json"
        self.vocabularies = load_vocabularies(vocab_path)

        # Load model
        logger.info(f"Loading KOGUT model from {self.model_path}")
        self.model = load_kogut_model(
            str(self.model_path),
            num_entities=len(self.vocabularies['entities']),
            num_relations=len(self.vocabularies['relations']),
            hidden_dim=config.DEFAULT_HIDDEN_DIM,
            device=self.device
        )

        # Create encoder
        self.encoder = KOGUTFeatureEncoder(
            self.model,
            self.vocabularies,
            device=self.device
        )

        logger.info("MicroGrowPredictor initialized successfully")

    def _validate_paths(self):
        """Validate that required files exist."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        vocab_path = self.data_path / "kogut" / "vocabularies.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabularies file not found: {vocab_path}")

    def predict(
        self,
        feature_string: str,
        topk: int = 20,
        hidden_dim: int = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Make predictions for given features.

        Args:
            feature_string: Comma-separated features (e.g., "temperature:mesophilic,oxygen:aerobe")
            topk: Number of top predictions to return
            hidden_dim: Hidden dimension for model (not used, kept for compatibility)

        Returns:
            Tuple of (predictions DataFrame, log output string)
        """
        log_output = []
        log_output.append(f"Input features: {feature_string}")

        # Parse feature string
        features_dict = {}
        for feature in feature_string.split(","):
            feature = feature.strip()
            if ":" in feature:
                feat_type, feat_value = feature.split(":", 1)
                features_dict[feat_type.strip()] = feat_value.strip()

        log_output.append(f"Parsed {len(features_dict)} features")

        try:
            # Encode features
            feature_embedding = self.encoder.encode_features(features_dict)
            log_output.append(f"Feature embedding shape: {feature_embedding.shape}")

            # Predict media
            predictions = self.encoder.predict_media(feature_embedding, topk=topk)
            log_output.append(f"Generated {len(predictions)} predictions")

            # Add scores
            predictions = compute_softmax_and_logit(predictions)

            # Count valid features
            valid_count = sum(
                1 for ft, fv in features_dict.items()
                if f"{ft}:{fv}" in self.encoder.entity2id
            )

            predictions = add_confidence_annotations(
                predictions,
                len(features_dict),
                valid_count
            )

            log_output.append(f"Feature coverage: {valid_count}/{len(features_dict)}")

            # Convert to DataFrame
            df = pd.DataFrame(predictions)

            # Get labels for media
            if not df.empty:
                media_ids = df['object'].unique().tolist()
                labels = self._get_labels(media_ids)
                df['object_label'] = df['object'].map(lambda x: labels.get(x, ''))

            return df, "\n".join(log_output)

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            log_output.append(error_msg)
            logger.error(error_msg, exc_info=True)
            raise RuntimeError("\n".join(log_output)) from e

    def _get_labels(self, entity_ids):
        """
        Get labels for entity IDs from the KG nodes table.

        Args:
            entity_ids: List of entity IDs to get labels for

        Returns:
            Dictionary mapping entity_id -> label
        """
        if not entity_ids:
            return {}

        kg_nodes_path = config.MICROGROWLINK_DIR / "data" / "merged-kg_nodes.tsv"

        if not kg_nodes_path.exists():
            logger.warning(f"KG nodes file not found: {kg_nodes_path}")
            return {}

        con = duckdb.connect(':memory:')

        try:
            # Load the KG nodes
            con.execute(f"""
                CREATE TABLE kg_nodes AS
                SELECT * FROM read_csv_auto('{kg_nodes_path}', delim='\t', header=true)
            """)

            # Get labels for these entities
            entity_str = "', '".join(entity_ids)
            labels_query = f"""
            SELECT
                id as entity_id,
                name as label
            FROM kg_nodes
            WHERE id IN ('{entity_str}')
            """

            labels_df = con.execute(labels_query).df()

            # Convert to dictionary
            labels_dict = dict(zip(labels_df['entity_id'], labels_df['label']))

            return labels_dict

        finally:
            con.close()

    def format_predictions_for_display(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Format predictions for Gradio display.

        Args:
            predictions: Raw predictions DataFrame

        Returns:
            Formatted DataFrame for display
        """
        # Select and rename columns for display
        display_df = predictions.copy()

        # Get labels for media
        media_ids = display_df['object'].unique().tolist()
        labels = self._get_labels(media_ids)

        # Create display columns
        display_df['Rank'] = display_df['rank'].astype(int)
        display_df['Medium'] = display_df['object']
        display_df['Medium Label'] = display_df['object'].map(lambda x: labels.get(x, ''))
        display_df['Score'] = display_df['raw_score'].round(4)
        display_df['Probability'] = display_df['softmax'].round(4)
        display_df['Confidence_Score'] = display_df['logit'].round(4)
        display_df['Confidence'] = display_df['confidence']

        # Select columns for display
        display_columns = [
            'Rank', 'Medium', 'Medium Label', 'Score',
            'Probability', 'Confidence_Score', 'Confidence'
        ]

        return display_df[display_columns]


def quick_predict(
    feature_string: str,
    topk: int = 20,
    device: str = "cpu"
) -> Tuple[pd.DataFrame, str]:
    """
    Quick prediction function using default configuration.

    Args:
        feature_string: Comma-separated features
        topk: Number of predictions
        device: Device to use

    Returns:
        Tuple of (formatted predictions DataFrame, log output)
    """
    predictor = MicroGrowPredictor(device=device)
    predictions, log = predictor.predict(feature_string, topk=topk)
    formatted = predictor.format_predictions_for_display(predictions)

    return formatted, log
