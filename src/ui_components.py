"""
Gradio UI components for MicroGrowLinkService.
"""

import gradio as gr
import config


def create_feature_inputs():
    """
    Create Gradio input components for microbial features.

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    # Temperature
    components['temperature'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['temperature'],
        value="unknown",
        label="Temperature Preference",
        info="Select the organism's preferred temperature range",
        elem_classes=["trait-input"]
    )

    # Oxygen requirement
    components['oxygen'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['oxygen'],
        value="unknown",
        label="Oxygen Requirement",
        info="Select the organism's relationship with oxygen",
        elem_classes=["trait-input"]
    )

    # Gram stain
    components['gram_stain'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['gram_stain'],
        value="unknown",
        label="Gram Stain",
        info="Select the Gram stain result",
        elem_classes=["trait-input"]
    )

    # Cell shape
    components['cell_shape'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['cell_shape'],
        value="unknown",
        label="Cell Shape",
        info="Select the cell morphology",
        elem_classes=["trait-input"]
    )

    # pH range
    components['ph_range'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['pH_range'],
        value="unknown",
        label="pH Range",
        info="Select the pH preference (low=acidophilic, mid=neutral, high=alkaliphilic)",
        elem_classes=["trait-input"]
    )

    # NaCl range
    components['nacl_range'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['NaCl_range'],
        value="unknown",
        label="NaCl Range",
        info="Select the salinity/NaCl preference",
        elem_classes=["trait-input"]
    )

    # Isolation Source - Theme selector
    isolation_themes = [None, "unknown"] + list(config.ISOLATION_SOURCE_HIERARCHY.keys())
    components['isolation_source_theme'] = gr.Dropdown(
        choices=isolation_themes,
        value="unknown",
        label="Isolation Source - Category",
        info="Select the broad category of isolation source",
        elem_classes=["trait-input"]
    )

    # Isolation Source - Specific source
    components['isolation_source'] = gr.Dropdown(
        choices=[None, "unknown"],
        value="unknown",
        label="Isolation Source - Specific",
        info="Select the specific isolation source (choose category first)",
        elem_classes=["trait-input"]
    )

    return components


def update_isolation_sources(theme):
    """
    Update isolation source choices based on selected theme.

    Args:
        theme: Selected theme

    Returns:
        Updated Dropdown component
    """
    if not theme or theme == "unknown":
        return gr.Dropdown(choices=[None, "unknown"], value="unknown")

    # Get sources for this theme
    theme_sources = config.ISOLATION_SOURCE_HIERARCHY.get(theme, [])
    source_choices = [None, "unknown"] + [item['value'] for item in theme_sources]

    return gr.Dropdown(choices=source_choices, value="unknown")


def create_advanced_inputs():
    """
    Create Gradio input components for advanced options.

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    # Top-K slider
    components['topk'] = gr.Slider(
        minimum=5,
        maximum=100,
        value=config.DEFAULT_TOPK,
        step=5,
        label="Number of Predictions",
        info="Top-ranked media to return (5-100)"
    )

    # Similar taxa threshold
    components['similar_taxa_threshold'] = gr.Slider(
        minimum=0,
        maximum=100,
        value=50,
        step=10,
        label="Similar Taxa Threshold (%)",
        info="Minimum % of your input traits that must be present in taxa (100% = all traits must match)"
    )

    # Device selection
    components['device'] = gr.Radio(
        choices=["cpu", "cuda"],
        value=config.DEFAULT_DEVICE,
        label="Device",
        info="Computation device (use CUDA if GPU available)"
    )

    # Hidden dimension (for model)
    components['hidden_dim'] = gr.Number(
        value=config.DEFAULT_HIDDEN_DIM,
        label="Hidden Dimension",
        info=f"Model's hidden dimension (default: {config.DEFAULT_HIDDEN_DIM} for KOGUT)",
        precision=0,
        minimum=1,
        maximum=2048
    )

    return components


def create_taxon_lookup_components():
    """
    Create Gradio input components for taxon lookup.

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    # Taxon ID input
    components['taxon_id'] = gr.Textbox(
        label="Taxon ID",
        placeholder="Enter NCBITaxon ID (e.g., NCBITaxon:372072) or BacDive strain ID",
        info="Lookup traits for a known taxon from KG-Microbe",
        lines=1
    )

    # Lookup status message
    components['lookup_status'] = gr.Markdown(
        value="Enter a taxon ID and click **Lookup Taxon** to auto-populate traits."
    )

    return components


def create_output_components():
    """
    Create Gradio output components.

    Returns:
        Dictionary of Gradio components
    """
    components = {}

    # Validation message
    components['validation'] = gr.Markdown(
        label="Feature Validation",
        value="Select features and click **Predict** to see validation results."
    )

    # Assessment summary (HTML)
    components['assessment_summary'] = gr.HTML(
        label="Information Content Assessment",
        value="Click <strong>Assess Trait Profile</strong> to evaluate trait quality."
    )

    # Detailed assessment metrics table
    components['assessment_details'] = gr.Dataframe(
        headers=["Category", "Metric", "Value"],
        label="Detailed Assessment Metrics",
        interactive=False,
        wrap=True,
        elem_classes=["results-table"],
        visible=False
    )

    # Feature importance table
    components['feature_importance'] = gr.Dataframe(
        headers=["Rank", "Feature", "Importance Score", "P-value", "Significance"],
        label="Feature Importance Rankings",
        interactive=False,
        wrap=True,
        elem_classes=["results-table"],
        visible=False
    )

    # Predictions table
    components['predictions'] = gr.Dataframe(
        headers=["Rank", "Medium", "Medium Label", "Score", "Probability", "Confidence_Score", "Confidence"],
        label="Prediction Results",
        interactive=False,
        wrap=True,
        elem_classes=["results-table"]
    )

    # Similar taxa table (new)
    components['similar_taxa'] = gr.Dataframe(
        headers=["Taxon", "Similarity", "Media", "Taxon Label", "Isolation Source", "Traits Matched %", "Trait Profile"],
        label="Similar Taxa in KG-Microbe (sorted by similarity, highest to lowest)",
        interactive=False,
        wrap=True,
        elem_classes=["results-table"]
    )

    # Prediction log (collapsible)
    components['log'] = gr.Textbox(
        label="Prediction Log (Details)",
        lines=10,
        max_lines=20,
        visible=True
    )

    # Download file
    components['download'] = gr.File(
        label="Download Full Results",
        visible=False
    )

    return components


def create_examples():
    """
    Create example inputs for the interface.

    Returns:
        List of example input combinations
    """
    examples = [
        # Example 1: NCBITaxon:372072 (Cohaesibacter gelatinilyticus) - marine bacterium, grows on medium:514 (Marine Broth)
        [
            "mid2",                      # temp_opt (mesophilic ~25-30°C)
            "facultative_anaerobe",      # oxygen
            "mid2",                      # pH_opt (neutral pH)
            "low",                       # NaCl_opt (marine salinity)
            "aerobic_heterotrophy",      # energy_metabolism
            "chitin_degradation",        # carbon_cycling
            "unknown",                   # nitrogen_cycling
            "unknown",                   # sulfur_metal_cycling
            "Environmental",             # isolation_source_theme
            "marine",                    # isolation_source
            20,                          # topk
            50,                          # similar_taxa_threshold
            "cpu",                       # device
            64                           # hidden_dim (KOGUT default)
        ],
        # Example 2: Novel Arctic Psychrophile - cold-adapted aerobic bacterium
        [
            "very_low",                  # temp_opt (psychrophilic ~0-15°C)
            "aerobe",                    # oxygen
            "mid2",                      # pH_opt (neutral)
            "very_low",                  # NaCl_opt (low salinity)
            "aerobic_chemo_heterotrophy", # energy_metabolism
            "cellulose_degradation",     # carbon_cycling
            "unknown",                   # nitrogen_cycling
            "unknown",                   # sulfur_metal_cycling
            "Other",                     # isolation_source_theme
            "glacier",                   # isolation_source
            20,                          # topk
            50,                          # similar_taxa_threshold
            "cpu",                       # device
            64                           # hidden_dim
        ],
        # Example 3: Novel Hyperthermophilic Anaerobe - extreme heat-loving organism from volcanic environments
        [
            "high",                      # temp_opt (hyperthermophilic ~80-100°C)
            "anaerobe",                  # oxygen
            "low",                       # pH_opt (acidophilic)
            "mid",                       # NaCl_opt (moderate salinity)
            "fermentation",              # energy_metabolism
            "unknown",                   # carbon_cycling
            "nitrogen_fixation",         # nitrogen_cycling
            "sulfur_reduction",          # sulfur_metal_cycling
            "Environmental",             # isolation_source_theme
            "hydrothermal-vent",         # isolation_source
            20,                          # topk
            50,                          # similar_taxa_threshold
            "cpu",                       # device
            64                           # hidden_dim
        ],
        # Example 4: Novel Extreme Halophile - salt-loving alkaliphile from saline environments
        [
            "mid2",                      # temp_opt (mesophilic)
            "aerobe",                    # oxygen
            "high",                      # pH_opt (alkaliphilic)
            "high",                      # NaCl_opt (high salinity)
            "aerobic_heterotrophy",      # energy_metabolism
            "aromatic_compound_degradation", # carbon_cycling
            "unknown",                   # nitrogen_cycling
            "unknown",                   # sulfur_metal_cycling
            "Environmental",             # isolation_source_theme
            "non-marine-saline-and-alkaline", # isolation_source
            20,                          # topk
            50,                          # similar_taxa_threshold
            "cpu",                       # device
            64                           # hidden_dim
        ]
    ]

    return examples


def create_help_text():
    """
    Create help text for the interface.

    Returns:
        Markdown-formatted help text
    """
    return """
## How to Use

1. **Select Microbial Traits**: Choose characteristics from the dropdown menus above
   - Default is 'unknown' for all traits (which are skipped in prediction)
   - Select at least 3 known features for reliable predictions
   - More features generally improve prediction accuracy
   - You can leave traits as 'unknown' or set to None if not applicable

2. **Configure Advanced Options** (Optional):
   - Number of predictions to return (5-100)
   - Device: CPU or CUDA
   - Hidden dimension: Default is 64 for KOGUT model (change only if using different model)

3. **Click Predict**: The system will:
   - Validate your feature selection
   - Run the KOGUT model to predict optimal growth media
   - Display ranked predictions with confidence scores

4. **Interpret Results**:
   - **Rank**: Position in the prediction list (1 = best)
   - **Medium**: Knowledge graph ID for the predicted medium
   - **Score**: Raw model score (higher = better)
   - **Probability**: Normalized probability across all media
   - **Confidence Score**: Sigmoid-normalized confidence [0-1]
   - **Confidence**: Overall confidence level (high/medium/low)

## Tips

- Include features from multiple categories for best results
- High confidence predictions (≥0.8) are generally reliable
- Low coverage warnings indicate features not in the knowledge graph
- Check the prediction log for detailed validation information

## Example Profiles

The example profiles showcase diverse microbial niches:
- **Example 1**: *Cohaesibacter gelatinilyticus* (NCBITaxon:372072) - Marine bacterium, grows on medium:514 (Marine Broth)
- **Example 2**: Arctic Psychrophile - Novel cold-adapted aerobic bacterium from glacial environments
- **Example 3**: Hyperthermophilic Anaerobe - Novel extreme heat-loving organism from hydrothermal vents
- **Example 4**: Extreme Halophile - Novel salt-loving alkaliphile from saline/alkaline environments

## About the Model

This tool uses a **KOGUT (Knowledge Graph Universal Transformer)** model trained on the KG-Microbe knowledge graph,
which integrates microbial trait data with growth media information.
"""
