"""
Configuration settings for MicroGrowLinkService Gradio app.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
MICROGROWLINK_DIR = BASE_DIR.parent / "MicroGrowLink"  # Still used for KG files reference

# Model configuration
MODEL_PATH = BASE_DIR / "models" / "kogut_20251026_212314.pt"
MODEL_TYPE = "kogut"

# Data configuration
DATA_PATH = BASE_DIR / "data"

# Device configuration
DEFAULT_DEVICE = "cpu"  # Use CPU for web server; change to "cuda" if GPU available

# Prediction defaults
DEFAULT_TOPK = 20
DEFAULT_HIDDEN_DIM = 64  # KOGUT model's hidden dimension (verified from checkpoint)

# Knowledge Graph data for assessment
KG_NODES_FILE = MICROGROWLINK_DIR / "data" / "merged-kg_nodes.tsv"
KG_EDGES_FILE = MICROGROWLINK_DIR / "data" / "merged-kg_edges.tsv"

# Assessment configuration
MIN_MATCHING_TAXA = 10  # Minimum matching taxa required for reliable statistics
ASSESSMENT_SIGNIFICANCE_LEVEL = 0.05  # Alpha level for statistical tests
BOOTSTRAP_ITERATIONS = 1000  # Number of bootstrap samples for CI calculation

# Feature categories and valid values
FEATURE_CATEGORIES = {
    "temp_opt": [
        "very_low",
        "low",
        "mid1",
        "mid2",
        "mid3",
        "mid4",
        "high"
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
    "pH_opt": [
        "low",
        "mid1",
        "mid2",
        "high"
    ],
    "NaCl_opt": [
        "very_low",
        "low",
        "mid",
        "high"
    ],
    "energy_metabolism": [
        "aerobic_anoxygenic_phototrophy",
        "aerobic_chemo_heterotrophy",
        "aerobic_heterotrophy",
        "anoxygenic_photoautotrophy",
        "anoxygenic_photoautotrophy_hydrogen_oxidation",
        "anoxygenic_photoautotrophy_iron_oxidation",
        "anoxygenic_photoautotrophy_sulfur_oxidation",
        "autotrophy",
        "fermentation",
        "heterotrophic",
        "photoautotrophy",
        "photoheterotrophy"
    ],
    "carbon_cycling": [
        "aliphatic_non_methane_hydrocarbon_degradation",
        "aminoacid_degradation",
        "aromatic_compound_degradation",
        "aromatic_hydrocarbon_degradation",
        "carbon_monoxide_oxidation",
        "cellobiose_degradation",
        "cellulose_degradation",
        "chitin_degradation",
        "chlorocarbon_degradation",
        "hydrocarbon_degradation",
        "methane_oxidation",
        "methanol_oxidation"
    ],
    "nitrogen_cycling": [
        "annamox",
        "denitrification",
        "nitrate_reduction_to_ammonia",
        "nitrification",
        "nitrite_reduction_to_ammonia",
        "nitrogen_fixation"
    ],
    "sulfur_metal_cycling": [
        "arsenate_reduction",
        "arsenite_oxidation",
        "iron_oxidation",
        "iron_reduction",
        "manganese_oxidation",
        "manganese_reduction",
        "sulfide_oxidation",
        "sulfur_oxidation",
        "sulfur_reduction"
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

    # Check KG data files for assessment
    if not KG_NODES_FILE.exists():
        errors.append(f"KG nodes file not found: {KG_NODES_FILE}")
    if not KG_EDGES_FILE.exists():
        errors.append(f"KG edges file not found: {KG_EDGES_FILE}")

    return errors
