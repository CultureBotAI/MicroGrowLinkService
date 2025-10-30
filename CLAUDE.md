# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MicroGrowLinkService is a Gradio web application for predicting optimal growth media for microorganisms based on their traits. It's part of the KG-Microbe project under KG-Hub and uses a KOGUT (Relational Graph Transformer) model trained on the KG-Microbe knowledge graph.

Users input microbial characteristics (temperature preference, oxygen requirement, Gram stain, cell shape, motility, sporulation) and receive ranked predictions of growth media with confidence scores.

## Architecture

### Application Structure

```
MicroGrowLinkService/
├── app.py                    # Main Gradio application entry point
├── config.py                 # Configuration (model/data paths, constants)
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py
│   ├── feature_utils.py     # Feature validation and parsing
│   ├── predict.py           # Prediction wrapper for MicroGrowLink
│   └── ui_components.py     # Gradio UI component definitions
└── data/                     # References MicroGrowLink/data/
```

### Component Responsibilities

**app.py**: Main entry point that:
- Creates Gradio interface with feature dropdowns and advanced options
- Wires up prediction function to UI components
- Handles initialization and error handling
- Launches web server on port 7860

**config.py**: Centralized configuration:
- Model path: Points to KOGUT .pt file in MicroGrowLink/models/
- Data path: Points to preprocessed data in MicroGrowLink/data/
- Feature categories: Valid values for each trait type
- Validation thresholds and UI constants

**src/feature_utils.py**: Feature handling:
- `build_feature_string()`: Converts dropdown selections to "type:value" format
  - Skips None and "unknown" values (treats them as not selected)
- `parse_feature_string()`: Parses feature strings into dictionaries
- `validate_features()`: Checks coverage against knowledge graph entities
- `format_validation_message()`: Creates user-friendly validation messages

**src/predict.py**: Prediction interface:
- `MicroGrowPredictor`: Wrapper class that calls MicroGrowLink's predict_novel_taxon.py
- Uses MicroGrowLink's `.venv/bin/python` directly (not uv)
- Runs command: `.venv/bin/python -m src.learn.predict_novel_taxon`
- Executes from MicroGrowLink directory (cwd) for proper imports
- The `-m` flag ensures proper Python package imports without PYTHONPATH issues
- Formats raw predictions into display-friendly DataFrames
- Handles temporary file management

**src/ui_components.py**: UI definitions:
- Feature input components (6 dropdowns for traits)
  - Each dropdown includes None, "unknown", and valid trait values
  - Default value is "unknown" (which is skipped when building feature string)
- Advanced options (top-k, device, hidden_dim)
- Output components (validation, predictions table, log)
- Example profiles for quick testing

### Data Flow

1. User selects microbial traits from dropdowns
2. `build_feature_string()` converts selections to "temperature:mesophilic,oxygen:aerobe" format
3. `validate_features()` checks features against entity2id.txt from knowledge graph
4. `MicroGrowPredictor.predict()` calls predict_novel_taxon.py via subprocess
5. Prediction script loads KOGUT model, encodes features, runs inference
6. Results parsed from TSV and formatted for display
7. User sees ranked media predictions with confidence scores

### External Dependencies

This service uses:

**Local Data** (in this repository):
- **Vocabularies**: `data/kogut/vocabularies.json` - Entity and relation vocabularies from KG-Microbe
- **Graph files**: `data/kogut/*.json` - Train/validation/test graph structures

**MicroGrowLink Repository** (sibling directory):
- **Model file**: `../MicroGrowLink/models/kogut_large_kg_20251024_143743_RESUME_20251026_212314.pt`
- **Prediction script**: `../MicroGrowLink/src/learn/predict_novel_taxon.py`

The MicroGrowLink repository must be present at `../MicroGrowLink/` relative to this service.

### Feature Validation

Features are validated before prediction:
- **Minimum 3 features recommended** (warns if less)
- **Minimum 2 categories recommended** (warns if less)
- **Minimum 50% coverage required** (errors if less)
- Coverage checked against entity2id.txt to ensure features exist in KG

Confidence levels:
- **High**: ≥80% coverage, ≥5 features, no warnings
- **Medium**: ≥60% coverage, ≥3 features
- **Low**: <60% coverage or <3 features

## Development Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Python 3.9+ (managed by uv via .python-version)
- MicroGrowLink repository cloned at `../MicroGrowLink/`
  - MicroGrowLink must have its own environment set up with PyTorch, models, etc.
  - This service uses subprocess to call MicroGrowLink's prediction script

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates venv automatically)
# Only installs Gradio and Pandas - minimal web service dependencies
uv sync

# Verify paths in config.py point to correct locations
uv run python -c "import config; print('\n'.join(config.validate_paths()) or 'Paths OK')"
```

**Important Architecture Decision**: This service has **minimal dependencies** (only Gradio and Pandas) because:
- It calls MicroGrowLink's prediction script via subprocess
- The prediction script runs in MicroGrowLink's environment with all the deep learning dependencies
- This keeps the web service lightweight and avoids complex PyTorch dependency issues
- All model loading, inference, and computation happens in the subprocess

### Running the Service

```bash
# Launch Gradio app with uv
uv run python app.py

# Or shorter
uv run app.py

# Access at http://localhost:7860
```

### Development Commands

```bash
# Check configuration
uv run python -c "import config; config.validate_paths()"

# Test feature validation
uv run python -c "from src.feature_utils import *; print(build_feature_string(temperature='mesophilic', oxygen='aerobe'))"

# Test prediction (requires valid model/data)
uv run python -c "from src.predict import quick_predict; print(quick_predict('temperature:mesophilic,oxygen:aerobe'))"

# Add a new dependency
uv add <package-name>

# Remove a dependency
uv remove <package-name>

# Update all dependencies
uv sync --upgrade
```

### Project Files

**pyproject.toml**: Modern Python project configuration
- Defines project metadata and dependencies
- Used by uv for package management
- Includes dev dependencies (pytest, black, ruff)

**.python-version**: Specifies Python version (3.11)
- Used by uv to select Python interpreter
- Ensures consistent Python version across environments

**requirements.txt**: Legacy dependency list (kept for compatibility)
- Can be used with pip if uv is not available
- Generated from pyproject.toml

## Key Implementation Notes

### Model Configuration

The KOGUT model requires `hidden_dim=64`:
- This was verified by inspecting the model checkpoint's `relation_embedding.weight` shape: `[24, 64]`
- The default is set in `config.DEFAULT_HIDDEN_DIM = 64`
- Users can override this in the UI if using a different model variant

### Subprocess vs Direct Import

The current implementation uses **subprocess** to call predict_novel_taxon.py rather than direct imports. This approach:
- **Pros**: Simpler, no need to copy complex prediction logic, easier to maintain
- **Cons**: Slower due to model reload per prediction, higher memory overhead

For production, consider refactoring to load the model once and import prediction functions directly.

### Model Caching

The predictor is initialized globally in app.py to avoid reloading. However, the subprocess approach still reloads the model for each prediction. To optimize:
1. Import prediction modules directly instead of subprocess
2. Load model once at app startup
3. Reuse loaded model for all predictions

### Error Handling

- Configuration errors shown at startup but don't block app launch
- Feature validation errors displayed to user before prediction
- Prediction failures caught and displayed with helpful messages
- Temporary TSV files cleaned up after each prediction

### Extending Features

To add new feature categories:
1. Add to `FEATURE_CATEGORIES` in config.py
2. Create dropdown in `create_feature_inputs()` in ui_components.py
3. Add parameter to `build_feature_string()` in feature_utils.py
4. Update examples in `create_examples()` in ui_components.py

### Model Selection

Currently hardcoded to KOGUT. To support multiple models:
1. Add model_type radio button to UI
2. Update config.py with paths for each model
3. Pass model_type to MicroGrowPredictor
4. Update predict_novel_taxon.py call with --model_type flag
