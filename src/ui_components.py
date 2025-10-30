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

    # Motility
    components['motility'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['motility'],
        value="unknown",
        label="Motility",
        info="Select the motility status",
        elem_classes=["trait-input"]
    )

    # Sporulation
    components['sporulation'] = gr.Dropdown(
        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['sporulation'],
        value="unknown",
        label="Sporulation",
        info="Select the sporulation capability",
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
        headers=["Taxon", "Taxon Label", "Isolation Source", "Traits Matched", "Traits Matched %", "Trait Profile", "Media Count", "Media (sample)"],
        label="Similar Taxa in KG-Microbe (with shared trait profiles and their media)",
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
        # Example 1: NCBITaxon:287 (Pseudomonas aeruginosa) - grows on medium:514
        [
            "mesophilic",           # temperature
            "aerobe",               # oxygen
            "negative",             # gram_stain
            "rod",                  # cell_shape
            "unknown",              # motility
            "unknown",              # sporulation
            "unknown",              # isolation_source
            20,                     # topk
            50,                     # similar_taxa_threshold
            "cpu",                  # device
            64                      # hidden_dim (KOGUT default)
        ],
        # Example 2: NCBITaxon:1931 (Streptomyces sp.) - grows on medium:65
        [
            "mesophilic",
            "aerobe",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            20,
            50,
            "cpu",
            64
        ],
        # Example 3: NCBITaxon:1502 (Clostridium perfringens) - anaerobic spore-former
        [
            "mesophilic",
            "anaerobe",
            "unknown",
            "rod",
            "unknown",
            "unknown",
            "unknown",
            20,
            50,
            "cpu",
            64
        ],
        # Example 4: NCBITaxon:459347 (Solibacillus cecembensis) - psychrophilic aerobe
        [
            "psychrophilic",
            "aerobe",
            "positive",
            "rod",
            "unknown",
            "unknown",
            "unknown",
            20,
            50,
            "cpu",
            64
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

The example profiles are based on real taxa from KG-Microbe:
- **Example 1**: NCBITaxon:287 (*Pseudomonas aeruginosa*) - grows on medium:514
- **Example 2**: NCBITaxon:1931 (*Streptomyces* sp.) - grows on medium:65
- **Example 3**: NCBITaxon:1502 (*Clostridium perfringens*) - anaerobic spore-former
- **Example 4**: NCBITaxon:459347 (*Solibacillus cecembensis*) - psychrophilic aerobe

## About the Model

This tool uses a **KOGUT (Knowledge Graph Universal Transformer)** model trained on the KG-Microbe knowledge graph,
which integrates microbial trait data with growth media information.
"""
