"""
Configuration settings for MicroGrowLinkService Gradio app.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
MICROGROWLINK_DIR = BASE_DIR.parent / "MicroGrowLink"

# Model configuration
MODEL_PATH = MICROGROWLINK_DIR / "models" / "kogut_20251026_212314.pt"
MODEL_TYPE = "kogut"

# Data configuration
# Use local data directory - the prediction script will add model-specific subdirectory (e.g., kogut/)
DATA_PATH = BASE_DIR / "data"

# Device configuration
DEFAULT_DEVICE = "cpu"  # Use CPU for web server; change to "cuda" if GPU available

# Prediction defaults
DEFAULT_TOPK = 20
DEFAULT_HIDDEN_DIM = 64  # KOGUT model's hidden dimension (verified from checkpoint)

# Feature categories and valid values
FEATURE_CATEGORIES = {
    "temperature": [
        "mesophilic",
        "thermophilic",
        "psychrophilic",
        "hyperthermophilic"
    ],
    "oxygen": [
        "aerobe",
        "anaerobe",
        "facultative_aerobe",
        "facultative_anaerobe",
        "microaerophile",
        "microaerotolerant",
        "obligate_aerobe",
        "obligate_anaerobe",
        "aerotolerant"
    ],
    "gram_stain": [
        "positive",
        "negative",
        "variable"
    ],
    "cell_shape": [
        "rod",
        "coccus",
        "bacillus",
        "spiral",
        "spirochete",
        "filament",
        "vibrio",
        "coccobacillus",
        "diplococcus",
        "oval",
        "ovoid",
        "sphere",
        "curved",
        "helical",
        "pleomorphic",
        "fusiform",
        "spindle"
    ],
    "motility": [
        "motile",
        "non_motile"
    ],
    "sporulation": [
        "spore_forming",
        "non_spore_forming"
    ]
}

# Isolation source hierarchy (loaded from JSON)
import json
ISOLATION_SOURCE_HIERARCHY_PATH = BASE_DIR / "data" / "isolation_source_hierarchy.json"

def load_isolation_source_hierarchy():
    """Load the isolation source hierarchy from JSON file."""
    if ISOLATION_SOURCE_HIERARCHY_PATH.exists():
        with open(ISOLATION_SOURCE_HIERARCHY_PATH, 'r') as f:
            return json.load(f)
    return {}

ISOLATION_SOURCE_HIERARCHY = load_isolation_source_hierarchy()

# Validation thresholds
MIN_FEATURES_RECOMMENDED = 3
MIN_CATEGORIES_RECOMMENDED = 2
MIN_COVERAGE_REQUIRED = 0.5  # 50%

# UI configuration
APP_TITLE = "MicroGrowLink: Microbial Growth Media Predictor"
APP_DESCRIPTION = """
Predict optimal growth media for microorganisms based on their traits.
Select microbial characteristics below and click **Predict** to get media recommendations.
"""

def validate_paths():
    """Validate that required paths exist."""
    errors = []

    if not MODEL_PATH.exists():
        errors.append(f"Model file not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        errors.append(f"Data directory not found: {DATA_PATH}")

    # For KOGUT model, check for vocabularies.json in kogut subdirectory
    if MODEL_TYPE == "kogut":
        vocab_path = DATA_PATH / "kogut" / "vocabularies.json"
        if not vocab_path.exists():
            errors.append(f"vocabularies.json not found in {DATA_PATH / 'kogut'}")
    elif MODEL_TYPE == "rotate":
        # For RotatE, check for entity2id.txt and relation2id.txt in rotate subdirectory
        entity2id_path = DATA_PATH / "rotate" / "entity2id.txt"
        if not entity2id_path.exists():
            errors.append(f"entity2id.txt not found in {DATA_PATH / 'rotate'}")

        relation2id_path = DATA_PATH / "rotate" / "relation2id.txt"
        if not relation2id_path.exists():
            errors.append(f"relation2id.txt not found in {DATA_PATH / 'rotate'}")
    elif MODEL_TYPE == "ultra":
        # For ULTRA, check in ultra subdirectory
        entity2id_path = DATA_PATH / "ultra" / "entity2id.txt"
        if not entity2id_path.exists():
            errors.append(f"entity2id.txt not found in {DATA_PATH / 'ultra'}")

    return errors
