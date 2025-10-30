# MicroGrowLinkService

A Gradio web application for predicting optimal growth media for microorganisms based on their traits, powered by the MicroGrowLink KOGUT model and KG-Microbe knowledge graph.

## Overview

MicroGrowLinkService provides an intuitive web interface where users can:
- Select microbial characteristics (temperature, oxygen requirement, Gram stain, cell shape, motility, sporulation, isolation source)
- Get ranked predictions of optimal growth media with confidence scores
- View similar taxa from KG-Microbe that share trait profiles
- Explore hierarchically organized isolation sources (352 environmental contexts)
- Download detailed results and validation feedback

The service uses a **KOGUT (Relational Graph Transformer)** model trained on the KG-Microbe knowledge graph to make predictions based on learned patterns in microbial growth data.

## Features

### Core Functionality
- **User-Friendly Interface**: Dropdown menus for easy feature selection with dark theme styling
- **7 Trait Categories**: Temperature, oxygen, Gram stain, cell shape, motility, sporulation, isolation source
- **Hierarchical Isolation Sources**: 352 sources organized into 6 themes (Host-Associated, Environmental, Medical/Clinical, Laboratory/Engineered, Food/Agriculture, Other)
- **Similar Taxa Finder**: Discover taxa with shared trait profiles and their associated media using Hamming distance
- **Smart Validation**: Real-time validation with coverage checks against 1.3M+ KG entities
- **Confidence Scoring**: Multiple confidence metrics (raw score, probability, logit)
- **Advanced Options**: Customizable prediction parameters and similarity thresholds
- **Real-World Examples**: Pre-loaded profiles based on actual taxa from KG-Microbe
- **Detailed Logging**: Full prediction logs for transparency

### Output Tables
1. **Prediction Results**: Ranked media with labels, scores, probabilities, and confidence levels
2. **Similar Taxa**: Taxa sharing your trait profile with their isolation sources, media, and match percentages

## Quick Start

### Prerequisites

1. **Python 3.9+** with [uv](https://github.com/astral-sh/uv) package manager (recommended)
2. **MicroGrowLink repository** cloned as a sibling directory at `../MicroGrowLink/`
3. **MicroGrowLink environment** with PyTorch and dependencies installed
4. **Model file**: Trained KOGUT model (`.pt` file)
5. **Knowledge graph data**: Nodes and edges TSV files

### Directory Structure (Expected)

```
parent_directory/
├── MicroGrowLink/                  # Core ML repository (cloned + data extracted)
│   ├── .venv/                      # Python virtual environment with PyTorch
│   │   └── bin/python              # Python interpreter used for predictions
│   ├── models/
│   │   └── kogut_20251026_212314.pt  # Trained KOGUT model (~150MB) [from Google Drive]
│   ├── data/
│   │   ├── merged-kg_edges.tsv     # KG edges (361MB) [from kgm_data.zip]
│   │   ├── merged-kg_nodes.tsv     # KG nodes (233MB) [from kgm_data.zip]
│   │   └── kogut/                  # KOGUT model data [from kogut_data.zip]
│   │       ├── vocabularies.json   # Entity→ID mappings (1.3M entities, ~45MB)
│   │       ├── train_graph.json    # Training graph structure
│   │       ├── val_graph.json      # Validation graph
│   │       ├── test_graph.json     # Test graph
│   │       └── node_features.json  # Node feature vectors
│   └── src/
│       ├── __init__.py             # Must exist!
│       ├── learn/
│       │   ├── __init__.py         # Must exist!
│       │   └── predict_novel_taxon.py  # Prediction script
│       ├── utils/
│       │   └── __init__.py         # Must exist!
│       ├── eval/
│       │   └── __init__.py         # Must exist!
│       ├── predict/
│       │   └── __init__.py         # Must exist!
│       └── attic/
│           └── __init__.py         # Must exist!
│
└── MicroGrowLinkService/           # This web application (this repository)
    ├── .venv/                      # Separate lightweight venv (Gradio, pandas, duckdb)
    ├── app.py                      # Main Gradio application
    ├── config.py                   # Configuration (paths, model settings)
    ├── data/
    │   ├── kogut/
    │   │   └── vocabularies.json   # Copy of vocabularies (for feature validation)
    │   └── isolation_source_hierarchy.json  # 352 sources organized by 6 themes
    ├── scripts/
    │   └── build_isolation_source_hierarchy.py
    └── src/
        ├── feature_utils.py        # Feature building and validation
        ├── predict.py              # Subprocess wrapper for predictions
        ├── similar_taxa.py         # Find similar taxa in KG
        └── ui_components.py        # Gradio UI components
```

## Installation

### Overview

This installation process involves:
1. Setting up the **MicroGrowLink** ML environment (PyTorch, heavy dependencies)
2. Downloading **3 files from Google Drive** (~615MB total):
   - `kgm_data.zip` - Knowledge graph data (nodes and edges)
   - `kogut_data.zip` - Model supporting data (vocabularies, graph structures)
   - `kogut_20251026_212314.pt` - Trained KOGUT model weights
3. Setting up **MicroGrowLinkService** web app (Gradio, lightweight dependencies)

**Total disk space required:** ~2.5GB (1.5GB for data, ~500MB for ML dependencies, ~500MB for environments)

**Estimated time:** 15-30 minutes (depending on download speeds)

### Step 1: Install uv Package Manager

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your shell or run:
source $HOME/.cargo/env
```

### Step 2: Set Up MicroGrowLink (Core ML Repository)

```bash
# Clone MicroGrowLink if not already done
cd parent_directory
git clone https://github.com/realmarcin/MicroGrowLink.git
cd MicroGrowLink

# Create __init__.py files in all src directories (CRITICAL!)
touch src/__init__.py
touch src/learn/__init__.py
touch src/utils/__init__.py
touch src/eval/__init__.py
touch src/predict/__init__.py
touch src/attic/__init__.py

# Create virtual environment and install dependencies
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Verify PyTorch installation
.venv/bin/python -c "import torch; print('✓ PyTorch', torch.__version__)"

# Verify imports work
.venv/bin/python -c "import src.learn.predict_novel_taxon; print('✓ Module imports working')"
```

### Step 3: Download Required Data Files from Google Drive

All required data files and the trained KOGUT model are available on Google Drive:

**Google Drive Folder:** https://drive.google.com/drive/folders/1mWGgYnyQiyMFIdotss4NPYdhfoUSITRi

**Required Files:**
- `kgm_data.zip` (~420MB compressed) - KG-Microbe knowledge graph (nodes & edges)
- `kogut_data.zip` (~45MB compressed) - KOGUT model supporting data (vocabularies, etc.)
- `kogut_20251026_212314.pt` (~150MB) - Trained KOGUT model weights

#### Option 1: Browser Download

1. Visit the Google Drive folder: https://drive.google.com/drive/folders/1mWGgYnyQiyMFIdotss4NPYdhfoUSITRi
2. Download all three files to a temporary directory (e.g., `~/Downloads`)
3. Proceed to the extraction steps below

#### Option 2: Command Line Download (using gdown)

```bash
# Install gdown if not already installed
pip install gdown

# Create temporary download directory
mkdir -p ~/Downloads/microgrowlink_data
cd ~/Downloads/microgrowlink_data

# Download KG-Microbe data (merged-kg_nodes.tsv, merged-kg_edges.tsv)
gdown "https://drive.google.com/uc?id=1b8d5aTlMvL-gANxHGwMZsZlI6J0jkD0-" -O kgm_data.zip

# Download KOGUT supporting data (vocabularies.json, etc.)
gdown "https://drive.google.com/uc?id=1rc17Xeh1JR-GPz81rc9PQqQQeYrzU7w9" -O kogut_data.zip

# Download KOGUT model file
gdown "https://drive.google.com/uc?id=1CQV7dVPnKHqG39zER6OKvn-x_WWevRkf" -O kogut_20251026_212314.pt

# Verify downloads
ls -lh kgm_data.zip kogut_data.zip kogut_20251026_212314.pt
```

#### Extract and Set Up Data Files

**Note:** Replace `~/Downloads/microgrowlink_data` with your actual download location if different.

```bash
# Navigate to MicroGrowLink directory
cd /path/to/parent_directory/MicroGrowLink

# 1. Extract KG-Microbe data to MicroGrowLink/data/
unzip ~/Downloads/microgrowlink_data/kgm_data.zip -d data/
# This creates:
#   data/merged-kg_edges.tsv (361MB)
#   data/merged-kg_nodes.tsv (233MB)

# 2. Extract KOGUT supporting data to MicroGrowLink/data/
unzip ~/Downloads/microgrowlink_data/kogut_data.zip -d data/
# This creates:
#   data/kogut/vocabularies.json
#   data/kogut/*.json (train/val/test graphs, node features)

# 3. Move KOGUT model to MicroGrowLink/models/
mkdir -p models
cp ~/Downloads/microgrowlink_data/kogut_20251026_212314.pt models/

# Verify all files are in place
echo "=== Verifying file structure ==="
ls -lh data/merged-kg_edges.tsv data/merged-kg_nodes.tsv
ls -lh data/kogut/vocabularies.json
ls -lh models/kogut_20251026_212314.pt

# Expected output:
# -rw-r--r-- 361M merged-kg_edges.tsv
# -rw-r--r-- 233M merged-kg_nodes.tsv
# -rw-r--r--  45M vocabularies.json
# -rw-r--r-- 150M kogut_20251026_212314.pt
```

**Expected directory structure after extraction:**
```
MicroGrowLink/
├── data/
│   ├── merged-kg_edges.tsv         # 361MB - KG relationships (taxon↔trait, taxon↔media)
│   ├── merged-kg_nodes.tsv         # 233MB - Entity labels and metadata
│   └── kogut/
│       ├── vocabularies.json       # 45MB - 1.3M entity→ID mappings
│       ├── train_graph.json        # Training graph structure
│       ├── val_graph.json          # Validation graph
│       ├── test_graph.json         # Test graph
│       └── node_features.json      # Node feature vectors
└── models/
    └── kogut_20251026_212314.pt    # 150MB - Trained KOGUT model weights
```

#### Verify Data Integrity

```bash
# Check entity count in vocabularies
python -c "import json; v=json.load(open('data/kogut/vocabularies.json')); print(f'✓ Loaded {len(v[\"entities\"]):,} entities and {len(v[\"relations\"]):,} relations')"
# Expected: ✓ Loaded 1,366,569 entities and 20 relations

# Check KG node/edge counts
wc -l data/merged-kg_*.tsv
# Expected: ~1.4M lines (edges), ~1.4M lines (nodes)
```

### Step 4: Set Up MicroGrowLinkService (This Web App)

```bash
cd ../MicroGrowLinkService

# Install dependencies using uv (creates lightweight .venv)
# Only installs Gradio, pandas, duckdb - no heavy ML dependencies
uv sync

# Verify installation
uv run python -c "import gradio; import pandas; import duckdb; print('✓ Dependencies installed')"
```

### Step 5: Prepare MicroGrowLinkService Data Files

#### Copy Vocabularies File
Copy the vocabularies file from MicroGrowLink to MicroGrowLinkService:

```bash
# Navigate to MicroGrowLinkService directory
cd /path/to/parent_directory/MicroGrowLinkService

# Create data directory structure
mkdir -p data/kogut

# Copy vocabularies.json from MicroGrowLink (extracted in Step 3)
cp ../MicroGrowLink/data/kogut/vocabularies.json data/kogut/

# Verify file exists and is valid JSON
ls -lh data/kogut/vocabularies.json
uv run python -c "import json; d=json.load(open('data/kogut/vocabularies.json')); print(f'✓ Loaded {len(d[\"entities\"]):,} entities and {len(d[\"relations\"]):,} relations')"
```

**Expected output:** `✓ Loaded 1,366,569 entities and 20 relations`

**Note**: The KOGUT model requires `vocabularies.json` for entity encoding. If you're using a different model format (RotatE, ULTRA), you'll need `entity2id.txt` instead and should update `MODEL_TYPE` in `config.py`.

**Why this file is needed in both locations:**
- **MicroGrowLink/data/kogut/vocabularies.json**: Used by the prediction script when encoding features
- **MicroGrowLinkService/data/kogut/vocabularies.json**: Used by the web app for feature validation (checking if features exist in the knowledge graph)

#### Build Isolation Source Hierarchy

```bash
# Generate hierarchical organization of 352 isolation sources
uv run python scripts/build_isolation_source_hierarchy.py

# Output:
# Extracting isolation sources from KG...
# Found 352 isolation sources
# Creating themed hierarchy...
# Host-Associated: 42 items
# Environmental: 40 items
# Medical/Clinical: 14 items
# Laboratory/Engineered: 10 items
# Food/Agriculture: 28 items
# Other: 218 items
# Saved hierarchy to data/isolation_source_hierarchy.json

# Verify file created
ls -lh data/isolation_source_hierarchy.json
```

### Step 6: Configure Paths

Edit `config.py` to match your setup:

```python
# Base directories
BASE_DIR = Path(__file__).parent
MICROGROWLINK_DIR = BASE_DIR.parent / "MicroGrowLink"  # Adjust if different location

# Model configuration
MODEL_PATH = MICROGROWLINK_DIR / "models" / "kogut_20251026_212314.pt"
MODEL_TYPE = "kogut"  # or "rotate", "ultra" if using different models

# Data configuration
DATA_PATH = BASE_DIR / "data"  # Contains kogut/ subdirectory with vocabularies.json

# Device configuration
DEFAULT_DEVICE = "cpu"  # Change to "cuda" if GPU available
DEFAULT_HIDDEN_DIM = 64  # KOGUT model's hidden dimension (verified from checkpoint)
```

### Step 7: Validate Configuration

```bash
# Run validation script
uv run python -c "
import config
errors = config.validate_paths()
if errors:
    print('❌ Configuration errors:')
    for e in errors:
        print(f'  - {e}')
else:
    print('✓ Configuration valid')
    print(f'  Model: {config.MODEL_PATH}')
    print(f'  Data: {config.DATA_PATH}')
    print(f'  Type: {config.MODEL_TYPE}')
"
```

## Running the Application

### Launch the Web Interface

```bash
# Start the Gradio app
uv run python app.py

# Or use the shorter command
uv run app.py
```

The application will:
1. Validate configuration
2. Display model and data paths
3. Create the Gradio interface
4. Launch at `http://localhost:7860` (or `http://0.0.0.0:7860`)

**Output:**
```
============================================================
MicroGrowLink: Microbial Growth Media Predictor
============================================================

Checking configuration...
✓ Configuration validated successfully

Settings:
  Model: .../MicroGrowLink/models/kogut_large_kg_*.pt
  Data:  .../MicroGrowLinkService/data
  Type:  kogut
  Device: cpu

Creating Gradio interface...
Launching application...
============================================================

Running on local URL:  http://127.0.0.1:7860
```

### Access the Interface

Open your browser and navigate to:
- **Local**: http://localhost:7860
- **Network**: http://0.0.0.0:7860 (accessible from other devices on your network)

## Usage Guide

### Basic Workflow

1. **Select Microbial Traits**
   - Choose from dropdown menus for each trait category
   - Default is `unknown` (skipped in prediction)
   - Recommended: Select at least 3-4 known traits for reliable predictions
   - More traits = better accuracy

2. **Optional: Select Isolation Source**
   - Choose a category (e.g., "Environmental", "Host-Associated")
   - Select specific source (e.g., "soil", "blood", "marine")
   - This hierarchical selector contains 352 sources from KG-Microbe

3. **Configure Advanced Options** (Optional - collapse accordion)
   - **Number of Predictions**: 5-100 (default: 20)
   - **Similar Taxa Threshold**: 0-100% of traits that must match (default: 50%)
   - **Device**: CPU or CUDA
   - **Hidden Dimension**: 64 for KOGUT (change only for different models)

4. **Click "🔬 Predict Growth Media"**

5. **Review Results**
   - **Feature Validation**: Coverage and warnings
   - **Prediction Results Table**: Ranked media with confidence scores
   - **Similar Taxa Table**: Taxa with shared trait profiles
   - **Detailed Log**: Full model output (collapsed)

### Example Profiles

Four real-world examples are pre-loaded:

#### Example 1: *Pseudomonas aeruginosa* (NCBITaxon:287)
- **Traits**: Mesophilic, aerobe, Gram-negative, rod-shaped
- **Grows on**: medium:514
- **Notes**: Common opportunistic pathogen, widely studied model organism

#### Example 2: *Streptomyces* sp. (NCBITaxon:1931)
- **Traits**: Mesophilic, aerobe
- **Grows on**: medium:65
- **Notes**: Antibiotic-producing actinobacterium

#### Example 3: *Clostridium perfringens* (NCBITaxon:1502)
- **Traits**: Mesophilic, anaerobe, rod-shaped
- **Notes**: Classic spore-former, food poisoning agent

#### Example 4: *Solibacillus cecembensis* (NCBITaxon:459347)
- **Traits**: Psychrophilic, aerobe, Gram-positive, rod-shaped
- **Notes**: Cold-loving bacterium from Antarctic environments

### Interpreting Results

#### Prediction Results Table
| Column | Description |
|--------|-------------|
| **Rank** | Position in prediction list (1 = best match) |
| **Medium** | Knowledge graph ID (e.g., `medium:514`) |
| **Medium Label** | Human-readable name from KG |
| **Score** | Raw model score (higher = better) |
| **Probability** | Softmax-normalized probability [0-1] |
| **Confidence_Score** | Sigmoid confidence [0-1] |
| **Confidence** | Level: high (≥0.8), medium (0.5-0.8), low (<0.5) |

#### Similar Taxa Table
| Column | Description |
|--------|-------------|
| **Taxon** | NCBI Taxonomy ID (e.g., `NCBITaxon:287`) |
| **Taxon Label** | Scientific name (e.g., *Pseudomonas aeruginosa*) |
| **Isolation Source** | Where the taxon was isolated from |
| **Traits Matched** | Number of traits that match your query |
| **Traits Matched %** | Percentage of YOUR traits present in this taxon |
| **Trait Profile** | Full trait profile as key:value pairs |
| **Media Count** | Number of media this taxon grows on |
| **Media (sample)** | Up to 5 media with labels |

**Note**: Similar taxa are sorted by similarity (100% = all shared traits match perfectly), then by % of your traits matched.

## Validation and Confidence

### Feature Validation

The app validates features before prediction:
- **Minimum 3 features recommended** (warns if less)
- **Minimum 2 categories recommended** (warns if less)
- **Minimum 50% coverage required** (blocks if less)

Coverage is checked against 1,366,569 entities in KG-Microbe to ensure features exist in the knowledge graph.

### Confidence Levels

Predictions are annotated with confidence based on:
- **Feature coverage**: What % of features are in KG
- **Number of features**: More features = higher confidence
- **Model scores**: Raw prediction confidence

| Level | Criteria |
|-------|----------|
| **High** | ≥80% coverage, ≥5 features, no warnings |
| **Medium** | ≥60% coverage, ≥3 features |
| **Low** | <60% coverage or <3 features |

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Browser                            │
│                   (Gradio Interface)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MicroGrowLinkService                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ app.py: Main Gradio application                      │  │
│  │  • UI event handlers                                 │  │
│  │  • Feature validation                                │  │
│  │  • Result formatting                                 │  │
│  └──────────────┬───────────────────────┬────────────────┘  │
│                 │                       │                   │
│    ┌────────────▼─────────┐   ┌────────▼──────────┐       │
│    │ src/predict.py       │   │ src/similar_taxa.py│       │
│    │ • Subprocess wrapper │   │ • DuckDB queries    │       │
│    │ • Label fetching     │   │ • Hamming distance  │       │
│    └────────────┬─────────┘   └─────────┬──────────┘       │
│                 │                       │                   │
│                 │ subprocess            │ SQL queries       │
│                 ▼                       ▼                   │
└─────────────────────────────────────────────────────────────┘
                  │                       │
    ┌─────────────▼───────────┐  ┌────────▼─────────────────┐
    │    MicroGrowLink        │  │  Knowledge Graph Files   │
    │  .venv/bin/python       │  │  • merged-kg_edges.tsv   │
    │  • PyTorch environment  │  │  • merged-kg_nodes.tsv   │
    │  • predict_novel_taxon  │  │  • 1.3M+ entities        │
    │  • KOGUT model          │  │  • 352 isolation sources │
    └─────────────────────────┘  └──────────────────────────┘
```

### Why Two Separate Environments?

**MicroGrowLinkService** (.venv):
- Lightweight: Gradio, pandas, duckdb (~50MB)
- Fast installation
- No GPU/CUDA dependencies
- Quick to update

**MicroGrowLink** (.venv):
- Heavy ML stack: PyTorch, torch-geometric, torch-scatter (~2-3GB)
- CUDA dependencies (if using GPU)
- Stable, pre-configured environment
- Only needed for predictions

This separation allows the web service to remain lightweight while leveraging the full ML capabilities of MicroGrowLink.

## Troubleshooting

### Installation Issues

#### "uv: command not found"
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### "ModuleNotFoundError: No module named 'src.utils'"
**Cause**: Missing `__init__.py` files in MicroGrowLink

**Fix**:
```bash
cd ../MicroGrowLink
touch src/__init__.py src/learn/__init__.py src/utils/__init__.py
touch src/eval/__init__.py src/predict/__init__.py src/attic/__init__.py

# Verify fix
.venv/bin/python -c "import src.learn.predict_novel_taxon; print('✓ Fixed')"
```

### Configuration Issues

#### "Model file not found"
**Check**:
```bash
ls -lh ../MicroGrowLink/models/*.pt
```
**Fix**: Update `MODEL_PATH` in `config.py` with correct filename

#### "vocabularies.json not found in data/kogut"
**Check**:
```bash
ls -lh data/kogut/vocabularies.json
```
**Fix**:
```bash
mkdir -p data/kogut
cp ../MicroGrowLink/data/kogut/vocabularies.json data/kogut/
```

#### "isolation_source_hierarchy.json not found"
**Fix**:
```bash
uv run python scripts/build_isolation_source_hierarchy.py
```

### Runtime Issues

#### "RuntimeError: size mismatch for relation_embedding.weight"
**Cause**: Hidden dimension mismatch between model and config

**Fix**: Inspect model to find correct dimension:
```bash
cd ../MicroGrowLink
.venv/bin/python << 'EOF'
import torch
model_path = "models/your_model.pt"
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print("Model state dict keys:")
for key, tensor in checkpoint['model_state_dict'].items():
    if 'embedding' in key:
        print(f"  {key}: {tensor.shape}")
EOF
```

Then update `DEFAULT_HIDDEN_DIM` in `config.py` to match (usually 64 for KOGUT).

#### "Prediction failed" or "Subprocess error"
**Debug**:
```bash
# Test prediction script directly
cd ../MicroGrowLink
.venv/bin/python -m src.learn.predict_novel_taxon \
  --features "temperature:mesophilic,oxygen:aerobe" \
  --model_type kogut \
  --model_path models/your_model.pt \
  --data_path ../MicroGrowLinkService/data \
  --output_file /tmp/test_predictions.tsv \
  --topk 10 \
  --device cpu \
  --hidden_dim 64

# Check output
cat /tmp/test_predictions.tsv
```

#### Low Coverage Warnings
**Cause**: Features may not exist in KG or use different naming

**Fix**: Check vocabularies.json for valid feature names:
```bash
uv run python << 'EOF'
import json
vocab = json.load(open('data/kogut/vocabularies.json'))
# Find all temperature features
temps = [e for e in vocab['entities'] if e.startswith('temperature:')]
print("Available temperature values:", temps)
EOF
```

#### CUDA Out of Memory
**Fix**: Switch to CPU mode in Advanced Options or update config:
```python
DEFAULT_DEVICE = "cpu"  # in config.py
```

### Performance Issues

#### Slow First Prediction
**Normal**: Model loading takes 10-30 seconds on first run. Subsequent predictions are faster (~1-5 seconds).

#### Slow Similar Taxa Query
**Normal**: DuckDB loads 361MB edges file into memory. First query takes 5-15 seconds. Subsequent queries are faster.

**Optimization**: Similar taxa queries can be pre-computed for common trait combinations.

## Development

### Running Tests

```bash
# Test feature validation
uv run python -c "
from src.feature_utils import build_feature_string, validate_features
import config
features = {'temperature': 'mesophilic', 'oxygen': 'aerobe'}
warnings, errors, coverage = validate_features(features, config.DATA_PATH)
print(f'Coverage: {coverage:.1%}')
print(f'Warnings: {warnings}')
print(f'Errors: {errors}')
"

# Test prediction wrapper
uv run python -c "
from src.predict import quick_predict
results, log = quick_predict('temperature:mesophilic,oxygen:aerobe')
print(results.head())
"

# Test similar taxa finder
uv run python -c "
from src.similar_taxa import find_similar_taxa
results = find_similar_taxa({'temperature': 'mesophilic', 'oxygen': 'aerobe'})
print(f'Found {len(results)} similar taxa')
"
```

### Adding New Features

To add a new microbial trait category:

1. **Update config.py**:
```python
FEATURE_CATEGORIES = {
    # ... existing categories ...
    "new_trait": ["value1", "value2", "value3"]
}
```

2. **Create UI component in src/ui_components.py**:
```python
components['new_trait'] = gr.Dropdown(
    choices=[None, "unknown"] + config.FEATURE_CATEGORIES['new_trait'],
    value="unknown",
    label="New Trait",
    info="Description of this trait",
    elem_classes=["trait-input"]
)
```

3. **Update feature_utils.py**:
```python
def build_feature_string(..., new_trait: str = None):
    # ...
    if new_trait and new_trait != "unknown":
        features.append(f"new_trait:{new_trait}")
```

4. **Update app.py**:
- Add parameter to `predict_media()`
- Add to inputs list in `predict_btn.click()`
- Update examples

### Custom Styling

Edit CSS in `app.py`:
```python
custom_css = """
/* Your custom styles */
"""
```

### Project Structure

```
MicroGrowLinkService/
├── app.py                          # Main Gradio application
├── config.py                       # Configuration (paths, categories)
├── requirements.txt                # Pip dependencies
├── pyproject.toml                  # uv/hatch project config
├── README.md                       # This file
├── CLAUDE.md                       # Developer documentation
├── LICENSE                         # License information
│
├── data/                           # Data files
│   ├── kogut/
│   │   └── vocabularies.json       # Entity vocabulary (1.3M entities)
│   └── isolation_source_hierarchy.json  # 352 sources by theme
│
├── scripts/                        # Utility scripts
│   └── build_isolation_source_hierarchy.py
│
└── src/                            # Source code
    ├── __init__.py
    ├── feature_utils.py            # Feature parsing & validation
    ├── predict.py                  # Prediction wrapper (subprocess)
    ├── similar_taxa.py             # Similar taxa finder (DuckDB)
    └── ui_components.py            # Gradio UI components
```

## Quick Reference

### Required Downloads from Google Drive

**Main Folder:** https://drive.google.com/drive/folders/1mWGgYnyQiyMFIdotss4NPYdhfoUSITRi

| File | Size | Description | Direct Download (gdown) |
|------|------|-------------|-------------------------|
| `kgm_data.zip` | ~420MB | KG-Microbe knowledge graph (merged-kg_edges.tsv, merged-kg_nodes.tsv) | `gdown "https://drive.google.com/uc?id=1b8d5aTlMvL-gANxHGwMZsZlI6J0jkD0-" -O kgm_data.zip` |
| `kogut_data.zip` | ~45MB | KOGUT model data (vocabularies.json, graph structures) | `gdown "https://drive.google.com/uc?id=1rc17Xeh1JR-GPz81rc9PQqQQeYrzU7w9" -O kogut_data.zip` |
| `kogut_20251026_212314.pt` | ~150MB | Trained KOGUT model weights | `gdown "https://drive.google.com/uc?id=1CQV7dVPnKHqG39zER6OKvn-x_WWevRkf" -O kogut_20251026_212314.pt` |

### File Placement After Extraction

```
MicroGrowLink/
├── data/
│   ├── merged-kg_edges.tsv        ← from kgm_data.zip
│   ├── merged-kg_nodes.tsv        ← from kgm_data.zip
│   └── kogut/
│       ├── vocabularies.json      ← from kogut_data.zip
│       └── *.json                 ← from kogut_data.zip
└── models/
    └── kogut_20251026_212314.pt   ← from Google Drive (direct download)

MicroGrowLinkService/
└── data/
    └── kogut/
        └── vocabularies.json      ← copied from MicroGrowLink/data/kogut/
```

### Key Configuration Values

**config.py:**
```python
MODEL_PATH = MICROGROWLINK_DIR / "models" / "kogut_20251026_212314.pt"
MODEL_TYPE = "kogut"
DATA_PATH = BASE_DIR / "data"  # Contains kogut/ subdirectory
DEFAULT_HIDDEN_DIM = 64  # KOGUT model hidden dimension
```

### Important Repositories

| Repository | URL | Purpose |
|------------|-----|---------|
| MicroGrowLinkService | https://github.com/realmarcin/MicroGrowLinkService | This web app |
| MicroGrowLink | https://github.com/realmarcin/MicroGrowLink | Core ML training/prediction |
| KG-Microbe | https://github.com/KG-Hub/KG-Microbe | Knowledge graph construction |

## Citation

If you use MicroGrowLinkService in your research, please cite:

```bibtex
@software{microgrowlinkservice2025,
  title = {MicroGrowLinkService: Web Interface for Microbial Growth Media Prediction},
  author = {KG-Microbe Team},
  year = {2025},
  url = {https://github.com/realmarcin/MicroGrowLinkService},
  note = {Powered by KOGUT model and KG-Microbe knowledge graph}
}
```

## Related Projects

- **[MicroGrowLink](https://github.com/realmarcin/MicroGrowLink)** - Core prediction models, training pipeline, and graph transformers
- **[KG-Microbe](https://github.com/KG-Hub/KG-Microbe)** - Microbial knowledge graph construction and integration
- **[KG-Hub](https://github.com/KG-Hub)** - Knowledge graph tools, resources, and best practices

## License

MIT License - See LICENSE file for details.

## Contact & Support

- **Issues**: https://github.com/realmarcin/MicroGrowLinkService/issues
- **Discussions**: https://github.com/realmarcin/MicroGrowLinkService/discussions
- **Email**: Contact the KG-Microbe team

## Acknowledgments

This work is supported by the KG-Hub initiative and leverages:
- KG-Microbe knowledge graph
- PyTorch and PyTorch Geometric
- Gradio for interactive interfaces
- DuckDB for efficient knowledge graph queries
- The microbiology and bioinformatics communities
