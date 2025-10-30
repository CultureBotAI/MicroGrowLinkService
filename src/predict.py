"""
Prediction wrapper for MicroGrowLink KOGUT model.
Calls the predict_novel_taxon.py script from MicroGrowLink.
"""

import subprocess
import tempfile
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Tuple
import config
import duckdb


class MicroGrowPredictor:
    """
    Wrapper for MicroGrowLink prediction functionality.
    """

    def __init__(self,
                 model_path: Path = None,
                 data_path: Path = None,
                 model_type: str = "kogut",
                 device: str = "cpu"):
        """
        Initialize predictor.

        Args:
            model_path: Path to model .pt file
            data_path: Path to data directory
            model_type: Model type ('kogut', 'rotate', or 'ultra')
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = model_path or config.MODEL_PATH
        self.data_path = data_path or config.DATA_PATH
        self.model_type = model_type
        self.device = device

        # Path to prediction script
        self.predict_script = config.MICROGROWLINK_DIR / "src" / "learn" / "predict_novel_taxon.py"

        # Validate paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required files exist."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        if not self.predict_script.exists():
            raise FileNotFoundError(f"Prediction script not found: {self.predict_script}")

    def predict(self,
                feature_string: str,
                topk: int = 20,
                hidden_dim: int = None) -> Tuple[pd.DataFrame, str]:
        """
        Make predictions for given features.

        Args:
            feature_string: Comma-separated features (e.g., "temperature:mesophilic,oxygen:aerobe")
            topk: Number of top predictions to return
            hidden_dim: Hidden dimension for model (None = use default from config)

        Returns:
            Tuple of (predictions DataFrame, stdout from prediction script)
        """
        # Use default hidden_dim if not provided
        if hidden_dim is None:
            hidden_dim = config.DEFAULT_HIDDEN_DIM
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as tmp:
            output_file = Path(tmp.name)

        try:
            # Use MicroGrowLink's existing .venv Python interpreter
            python_exe = config.MICROGROWLINK_DIR / ".venv" / "bin" / "python"

            # Fallback to system python if .venv doesn't exist
            if not python_exe.exists():
                python_exe = "python"

            # Build command - run as a module to ensure proper imports
            # Use -m flag to run as module: python -m src.learn.predict_novel_taxon
            cmd = [
                str(python_exe), "-m", "src.learn.predict_novel_taxon",
                "--features", feature_string,
                "--model_type", self.model_type,
                "--model_path", str(self.model_path),
                "--data_path", str(self.data_path),
                "--output_file", str(output_file),
                "--topk", str(topk),
                "--device", self.device,
                "--hidden_dim", str(hidden_dim)
            ]

            # Run prediction script in MicroGrowLink directory
            # Using MicroGrowLink's .venv ensures all dependencies are available
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(config.MICROGROWLINK_DIR)
            )

            # Read results
            if output_file.exists():
                predictions = pd.read_csv(output_file, sep='\t')
            else:
                raise RuntimeError("Prediction script did not generate output file")

            return predictions, result.stdout

        except subprocess.CalledProcessError as e:
            error_msg = f"Prediction failed:\n{e.stderr}"
            raise RuntimeError(error_msg) from e

        finally:
            # Clean up temporary file
            if output_file.exists():
                output_file.unlink()

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
        display_columns = ['Rank', 'Medium', 'Medium Label', 'Score', 'Probability', 'Confidence_Score', 'Confidence']

        # Add neighbor info if available
        if 'neighbor_support' in display_df.columns:
            display_df['Neighbor_Support'] = display_df['neighbor_support']
            display_columns.append('Neighbor_Support')

        if 'neighbor_score' in display_df.columns:
            display_df['Neighbor_Score'] = display_df['neighbor_score'].round(4)
            display_columns.append('Neighbor_Score')

        return display_df[display_columns]


def quick_predict(feature_string: str,
                  topk: int = 20,
                  device: str = "cpu") -> Tuple[pd.DataFrame, str]:
    """
    Quick prediction function using default configuration.

    Args:
        feature_string: Comma-separated features
        topk: Number of predictions
        device: Device to use

    Returns:
        Tuple of (formatted predictions DataFrame, validation log)
    """
    predictor = MicroGrowPredictor(device=device)
    predictions, log = predictor.predict(feature_string, topk=topk)
    formatted = predictor.format_predictions_for_display(predictions)

    return formatted, log
