"""
Feature validation and parsing utilities for MicroGrowLinkService.
Adapted from MicroGrowLink feature validation logic.
"""

from typing import Dict, List, Tuple, Set
from pathlib import Path
import json
import config


def build_feature_string(temperature: str = None,
                         oxygen: str = None,
                         gram_stain: str = None,
                         cell_shape: str = None,
                         motility: str = None,
                         sporulation: str = None,
                         isolation_source: str = None) -> str:
    """
    Build feature string from dropdown selections.

    Args:
        temperature: Temperature preference
        oxygen: Oxygen requirement
        gram_stain: Gram stain result
        cell_shape: Cell morphology
        motility: Motility status
        sporulation: Sporulation capability
        isolation_source: Isolation source value

    Returns:
        Comma-separated feature string (e.g., "temperature:mesophilic,oxygen:aerobe")
    """
    features = []

    # Skip None and 'unknown' values
    if temperature and temperature != "unknown":
        features.append(f"temperature:{temperature}")
    if oxygen and oxygen != "unknown":
        features.append(f"oxygen:{oxygen}")
    if gram_stain and gram_stain != "unknown":
        features.append(f"gram_stain:{gram_stain}")
    if cell_shape and cell_shape != "unknown":
        features.append(f"cell_shape:{cell_shape}")
    if motility and motility != "unknown":
        features.append(f"motility:{motility}")
    if sporulation and sporulation != "unknown":
        features.append(f"sporulation:{sporulation}")
    if isolation_source and isolation_source != "unknown":
        features.append(f"isolation_source:{isolation_source}")

    return ",".join(features)


def parse_feature_string(feature_string: str) -> Dict[str, str]:
    """
    Parse feature string into dictionary.

    Args:
        feature_string: Comma-separated features like "temperature:mesophilic,oxygen:aerobe"

    Returns:
        Dictionary mapping feature types to values
    """
    features = {}
    if not feature_string:
        return features

    for feature in feature_string.split(","):
        feature = feature.strip()
        if ":" in feature:
            feat_type, feat_value = feature.split(":", 1)
            features[feat_type.strip()] = feat_value.strip()

    return features


def load_known_entities(data_path: Path) -> Set[str]:
    """
    Load known entities from vocabularies.json (KOGUT) or entity2id.txt (RotatE/ULTRA).

    Args:
        data_path: Path to data directory (should contain model-specific subdirectories)

    Returns:
        Set of known entity IDs
    """
    known_entities = set()

    # Try vocabularies.json first (KOGUT format in kogut/ subdirectory)
    vocab_path = data_path / "kogut" / "vocabularies.json"
    if vocab_path.exists():
        try:
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                # Get entity IDs from the "entities" dictionary
                if "entities" in vocab_data:
                    known_entities = set(vocab_data["entities"].keys())
            return known_entities
        except Exception as e:
            print(f"Warning: Could not load entities from {vocab_path}: {e}")

    # Fall back to entity2id.txt (RotatE format in rotate/ subdirectory)
    entity2id_path = data_path / "rotate" / "entity2id.txt"
    if entity2id_path.exists():
        try:
            with open(entity2id_path, 'r') as f:
                lines = f.readlines()
                # Skip first line (count)
                for line in lines[1:]:
                    entity = line.strip().split('\t')[0]
                    known_entities.add(entity)
        except Exception as e:
            print(f"Warning: Could not load entities from {entity2id_path}: {e}")

    # Try ULTRA format in ultra/ subdirectory
    entity2id_path = data_path / "ultra" / "entity2id.txt"
    if not known_entities and entity2id_path.exists():
        try:
            with open(entity2id_path, 'r') as f:
                lines = f.readlines()
                # Skip first line (count)
                for line in lines[1:]:
                    entity = line.strip().split('\t')[0]
                    known_entities.add(entity)
        except Exception as e:
            print(f"Warning: Could not load entities from {entity2id_path}: {e}")

    return known_entities


def validate_features(features_dict: Dict[str, str],
                     data_path: Path) -> Tuple[List[str], List[str], float]:
    """
    Validate feature quality and coverage.

    Args:
        features_dict: Dictionary of feature type -> value
        data_path: Path to data directory

    Returns:
        Tuple of (warnings, errors, coverage_ratio)
    """
    warnings = []
    errors = []

    # Check minimum features
    num_features = len(features_dict)
    if num_features < config.MIN_FEATURES_RECOMMENDED:
        warnings.append(
            f"⚠ Only {num_features} feature(s) provided. "
            f"Recommended: ≥{config.MIN_FEATURES_RECOMMENDED} for better predictions."
        )

    # Check feature categories
    num_categories = len(set(features_dict.keys()))
    if num_categories < config.MIN_CATEGORIES_RECOMMENDED:
        warnings.append(
            f"⚠ Features from only {num_categories} category(ies). "
            f"Recommended: ≥{config.MIN_CATEGORIES_RECOMMENDED} for more accurate results."
        )

    # Load known entities and check coverage
    known_entities = load_known_entities(data_path)

    if known_entities:
        # Build full feature IDs
        feature_ids = []
        for feat_type, feat_value in features_dict.items():
            feature_id = f"{feat_type}:{feat_value}"
            feature_ids.append(feature_id)

        # Check which features exist in KG
        found_features = [f for f in feature_ids if f in known_entities]
        coverage = len(found_features) / len(feature_ids) if feature_ids else 0

        if coverage < config.MIN_COVERAGE_REQUIRED:
            errors.append(
                f"❌ Only {coverage*100:.0f}% of features found in knowledge graph. "
                f"Minimum required: {config.MIN_COVERAGE_REQUIRED*100:.0f}%. "
                f"Prediction may not be reliable."
            )
        elif coverage < 1.0:
            missing = set(feature_ids) - set(found_features)
            warnings.append(
                f"⚠ {len(missing)} feature(s) not found in KG: {', '.join(missing)}"
            )
    else:
        warnings.append("⚠ Could not verify feature coverage (vocabulary file not loaded)")
        coverage = 1.0  # Assume full coverage if can't check

    return warnings, errors, coverage


def format_validation_message(features_dict: Dict[str, str],
                               warnings: List[str],
                               errors: List[str],
                               coverage: float) -> str:
    """
    Format validation results into user-friendly message.

    Args:
        features_dict: Dictionary of features
        warnings: List of warning messages
        errors: List of error messages
        coverage: Feature coverage ratio (0-1)

    Returns:
        Formatted validation message
    """
    lines = []

    # Feature summary
    lines.append("### Feature Validation Summary")
    lines.append(f"- **Features provided**: {len(features_dict)}")
    lines.append(f"- **Categories**: {len(set(features_dict.keys()))}")
    lines.append(f"- **Coverage**: {coverage*100:.1f}%")
    lines.append("")

    # Feature list
    if features_dict:
        lines.append("### Selected Features")
        for feat_type, feat_value in features_dict.items():
            lines.append(f"- `{feat_type}`: {feat_value}")
        lines.append("")

    # Errors (blocking)
    if errors:
        lines.append("### ❌ Errors")
        for error in errors:
            lines.append(error)
        lines.append("")

    # Warnings (non-blocking)
    if warnings:
        lines.append("### ⚠ Warnings")
        for warning in warnings:
            lines.append(warning)
        lines.append("")

    # Success message
    if not errors and not warnings:
        lines.append("### ✓ Validation Passed")
        lines.append("All features are valid and coverage is excellent!")

    return "\n".join(lines)
