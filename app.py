"""
MicroGrowLink Service - Gradio Web Application
Predicts optimal growth media for microorganisms based on their traits.
"""

import gradio as gr
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.feature_utils import (
    build_feature_string,
    parse_feature_string,
    validate_features,
    format_validation_message
)
from src.predict import MicroGrowPredictor
from src.similar_taxa import find_similar_taxa, format_similar_taxa_for_display
from src.ui_components import (
    create_feature_inputs,
    create_advanced_inputs,
    create_output_components,
    create_examples,
    create_help_text,
    update_isolation_sources
)


# Initialize predictor (global to avoid reloading model)
predictor = None


def initialize_predictor(device="cpu"):
    """Initialize the predictor with error handling."""
    global predictor

    try:
        # Validate paths first
        errors = config.validate_paths()
        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(errors)
            return False, error_msg

        # Create predictor
        predictor = MicroGrowPredictor(device=device)
        return True, "Predictor initialized successfully"

    except Exception as e:
        return False, f"Failed to initialize predictor: {str(e)}"


def predict_media(temp_opt, oxygen, ph_opt, nacl_opt, energy_metabolism, carbon_cycling,
                  nitrogen_cycling, sulfur_metal_cycling, isolation_source, topk, similar_taxa_threshold, device, hidden_dim):
    """
    Main prediction function called by Gradio interface.

    Args:
        temp_opt: Optimal temperature
        oxygen: Oxygen requirement
        ph_opt: Optimal pH
        nacl_opt: Optimal NaCl/salinity
        energy_metabolism: Energy metabolism pathway
        carbon_cycling: Carbon cycling pathway
        nitrogen_cycling: Nitrogen cycling pathway
        sulfur_metal_cycling: Sulfur/metal cycling pathway
        isolation_source: Isolation source
        topk: Number of predictions
        similar_taxa_threshold: Minimum % of traits that must be matched in similar taxa
        device: Device to use
        hidden_dim: Model hidden dimension

    Returns:
        Tuple of (validation_message, predictions_df, similar_taxa_df, log)
    """
    global predictor

    try:
        # Re-initialize predictor if device changed or not initialized
        if predictor is None or predictor.device != device:
            success, msg = initialize_predictor(device)
            if not success:
                return msg, None, None, msg

        # Build feature string
        feature_string = build_feature_string(
            temp_opt=temp_opt,
            oxygen=oxygen,
            ph_opt=ph_opt,
            nacl_opt=nacl_opt,
            energy_metabolism=energy_metabolism,
            carbon_cycling=carbon_cycling,
            nitrogen_cycling=nitrogen_cycling,
            sulfur_metal_cycling=sulfur_metal_cycling,
            isolation_source=isolation_source
        )

        # Check if any features provided
        if not feature_string:
            error_msg = "❌ No features selected. Please select at least one microbial trait."
            return error_msg, None, None, ""

        # Parse features
        features_dict = parse_feature_string(feature_string)

        # Validate features
        warnings, errors, coverage = validate_features(features_dict, config.DATA_PATH)

        # Format validation message
        validation_msg = format_validation_message(features_dict, warnings, errors, coverage)

        # Check for blocking errors
        if errors:
            return validation_msg, None, None, ""

        # Make predictions
        predictions, log = predictor.predict(
            feature_string=feature_string,
            topk=topk,
            hidden_dim=hidden_dim
        )

        # Format predictions for display
        formatted_predictions = predictor.format_predictions_for_display(predictions)

        # Find similar taxa with shared trait profiles
        similar_taxa = find_similar_taxa(
            features_dict,
            top_k=10,
            min_traits_pct=similar_taxa_threshold
        )
        formatted_similar_taxa = format_similar_taxa_for_display(similar_taxa)

        return validation_msg, formatted_predictions, formatted_similar_taxa, log

    except Exception as e:
        error_msg = f"❌ Prediction failed: {str(e)}"
        return error_msg, None, None, str(e)


def create_interface():
    """
    Create the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    # Check configuration on startup
    config_errors = config.validate_paths()
    if config_errors:
        print("⚠ Configuration warnings:")
        for error in config_errors:
            print(f"  - {error}")
        print("\nThe app will start, but predictions may fail.")
        print("Please verify the model and data paths in config.py\n")

    # Create custom theme
    custom_theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="gray",
    ).set(
        # Trait profile dropdowns - black and dark gray
        input_background_fill="*neutral_950",
        input_background_fill_dark="*neutral_900",
        input_border_color="*neutral_700",
        input_border_color_dark="*neutral_600",
        # Table backgrounds - dark gray
        table_even_background_fill="*neutral_800",
        table_odd_background_fill="*neutral_850",
        table_border_color="*neutral_700",
    )

    # Custom CSS for additional styling
    custom_css = """
    /* White divider between left and right columns */
    .main-row > .left-column {
        border-right: 2px solid white !important;
        padding-right: 20px !important;
    }

    .main-row > .right-column {
        padding-left: 20px !important;
    }

    /* Results tables - dark gray background */
    .results-table {
        background-color: #374151 !important;
    }

    .results-table .table-wrap {
        background-color: #374151 !important;
    }

    .results-table table {
        background-color: #374151 !important;
    }

    .results-table thead {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
    }

    .results-table tbody tr:nth-child(even) {
        background-color: #4b5563 !important;
    }

    .results-table tbody tr:nth-child(odd) {
        background-color: #374151 !important;
    }

    .results-table tbody td {
        color: #e5e7eb !important;
        border-color: #6b7280 !important;
    }

    /* Trait profile inputs - very light orange background with bold black text */
    .trait-input label {
        color: white !important;
        background-color: #1e3a8a !important;
        font-weight: bold !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        display: inline-block !important;
        margin-bottom: 8px !important;
    }

    .trait-input .wrap {
        background-color: #ffd9b3 !important;
        border-color: #ffb366 !important;
        border-radius: 8px !important;
    }

    .trait-input input,
    .trait-input select,
    .trait-input .dropdown {
        background-color: #ffd9b3 !important;
        border-color: #ffb366 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }

    .trait-input .dropdown-arrow {
        color: #000000 !important;
    }

    /* Dropdown menu options - very light orange background */
    .trait-input option {
        background-color: #ffd9b3 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Dropdown menu container */
    .trait-input .dropdown-menu {
        background-color: #ffd9b3 !important;
    }

    .trait-input .dropdown-content {
        background-color: #ffd9b3 !important;
    }

    /* Selected dropdown items */
    .trait-input option:checked,
    .trait-input option:hover {
        background-color: #ffb366 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Green predict button */
    button.primary {
        background: linear-gradient(to bottom right, #22c55e, #16a34a) !important;
        border-color: #16a34a !important;
        color: white !important;
        font-weight: bold !important;
    }

    button.primary:hover {
        background: linear-gradient(to bottom right, #16a34a, #15803d) !important;
        border-color: #15803d !important;
    }
    """

    with gr.Blocks(title=config.APP_TITLE, theme=custom_theme, css=custom_css) as interface:
        # Header
        gr.Markdown(f"# {config.APP_TITLE}")
        gr.Markdown(config.APP_DESCRIPTION)

        with gr.Row(elem_classes=["main-row"]):
            with gr.Column(scale=1, elem_classes=["left-column"]):
                gr.Markdown("## Microbial Traits")
                gr.Markdown("Select characteristics of the microorganism:")

                # Create feature inputs dictionary
                feature_inputs = {}

                # Layout in 4×3 grid
                with gr.Row():
                    feature_inputs['temp_opt'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['temp_opt'],
                        value="unknown",
                        label="Optimal Temperature",
                        info="Select the organism's optimal temperature (very_low to high)",
                        elem_classes=["trait-input"]
                    )
                    feature_inputs['oxygen'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['oxygen'],
                        value="unknown",
                        label="Oxygen Requirement",
                        info="Select the organism's relationship with oxygen",
                        elem_classes=["trait-input"]
                    )
                    feature_inputs['ph_opt'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['pH_opt'],
                        value="unknown",
                        label="Optimal pH",
                        info="Select the optimal pH (low=acidophilic, high=alkaliphilic)",
                        elem_classes=["trait-input"]
                    )

                with gr.Row():
                    feature_inputs['nacl_opt'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['NaCl_opt'],
                        value="unknown",
                        label="Optimal NaCl",
                        info="Select the optimal salinity/NaCl level",
                        elem_classes=["trait-input"]
                    )
                    feature_inputs['energy_metabolism'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['energy_metabolism'],
                        value="unknown",
                        label="Energy Metabolism",
                        info="Select energy metabolism pathway (phototrophy, chemotrophy, etc.)",
                        elem_classes=["trait-input"]
                    )
                    feature_inputs['carbon_cycling'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['carbon_cycling'],
                        value="unknown",
                        label="Carbon Cycling",
                        info="Select carbon cycling pathway (degradation, oxidation, etc.)",
                        elem_classes=["trait-input"]
                    )

                with gr.Row():
                    feature_inputs['nitrogen_cycling'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['nitrogen_cycling'],
                        value="unknown",
                        label="Nitrogen Cycling",
                        info="Select nitrogen cycling pathway (fixation, nitrification, etc.)",
                        elem_classes=["trait-input"]
                    )
                    feature_inputs['sulfur_metal_cycling'] = gr.Dropdown(
                        choices=[None, "unknown"] + config.FEATURE_CATEGORIES['sulfur_metal_cycling'],
                        value="unknown",
                        label="Sulfur/Metal Cycling",
                        info="Select sulfur/metal cycling pathway (oxidation, reduction, etc.)",
                        elem_classes=["trait-input"]
                    )
                    isolation_themes = [None, "unknown"] + list(config.ISOLATION_SOURCE_HIERARCHY.keys())
                    feature_inputs['isolation_source_theme'] = gr.Dropdown(
                        choices=isolation_themes,
                        value="unknown",
                        label="Isolation Source - Category",
                        info="Select the broad category of isolation source",
                        elem_classes=["trait-input"]
                    )

                with gr.Row():
                    feature_inputs['isolation_source'] = gr.Dropdown(
                        choices=[None, "unknown"],
                        value="unknown",
                        label="Isolation Source - Specific",
                        info="Select the specific isolation source (choose category first)",
                        elem_classes=["trait-input"]
                    )

                # Wire up isolation source theme to update source dropdown
                feature_inputs['isolation_source_theme'].change(
                    fn=update_isolation_sources,
                    inputs=[feature_inputs['isolation_source_theme']],
                    outputs=[feature_inputs['isolation_source']]
                )

                # Advanced options in accordion
                with gr.Accordion("⚙ Advanced Options", open=False):
                    advanced_inputs = create_advanced_inputs()

                # Predict button
                predict_btn = gr.Button("🔬 Predict Growth Media", variant="primary", size="lg")

                # Examples
                gr.Markdown("### 📋 Example Profiles")
                gr.Examples(
                    examples=create_examples(),
                    inputs=[
                        feature_inputs['temp_opt'],
                        feature_inputs['oxygen'],
                        feature_inputs['ph_opt'],
                        feature_inputs['nacl_opt'],
                        feature_inputs['energy_metabolism'],
                        feature_inputs['carbon_cycling'],
                        feature_inputs['nitrogen_cycling'],
                        feature_inputs['sulfur_metal_cycling'],
                        feature_inputs['isolation_source_theme'],
                        feature_inputs['isolation_source'],
                        advanced_inputs['topk'],
                        advanced_inputs['similar_taxa_threshold'],
                        advanced_inputs['device'],
                        advanced_inputs['hidden_dim']
                    ],
                    label="Click an example to load it"
                )

                # Example descriptions
                gr.Markdown("""
**Example Descriptions:**
1. **Marine bacterium** - aerobic heterotrophy + chitin degradation
2. **Arctic psychrophile** - aerobic chemo-heterotrophy + cellulose degradation
3. **Hyperthermophilic anaerobe** - fermentation + nitrogen fixation + sulfur reduction
4. **Extreme halophile** - aerobic heterotrophy + aromatic compound degradation
                """)

            with gr.Column(scale=1, elem_classes=["right-column"]):
                gr.Markdown("## Prediction Results")

                # Create output components
                output_components = create_output_components()

                # Validation message
                validation_output = output_components['validation']

                # Predictions table
                predictions_output = output_components['predictions']

                # Similar taxa table
                similar_taxa_output = output_components['similar_taxa']

                # Log output (collapsed by default)
                with gr.Accordion("📋 Detailed Log", open=False):
                    log_output = output_components['log']

        # Help section
        with gr.Accordion("ℹ️ Help & Information", open=False):
            gr.Markdown(create_help_text())

        # Wire up the predict button
        predict_btn.click(
            fn=predict_media,
            inputs=[
                feature_inputs['temp_opt'],
                feature_inputs['oxygen'],
                feature_inputs['ph_opt'],
                feature_inputs['nacl_opt'],
                feature_inputs['energy_metabolism'],
                feature_inputs['carbon_cycling'],
                feature_inputs['nitrogen_cycling'],
                feature_inputs['sulfur_metal_cycling'],
                feature_inputs['isolation_source'],
                advanced_inputs['topk'],
                advanced_inputs['similar_taxa_threshold'],
                advanced_inputs['device'],
                advanced_inputs['hidden_dim']
            ],
            outputs=[
                validation_output,
                predictions_output,
                similar_taxa_output,
                log_output
            ]
        )

        # Footer
        gr.Markdown("---")
        gr.Markdown(
            "🧬 Powered by [MicroGrowLink](https://github.com/realmarcin/MicroGrowLink) | "
            f"Model: KOGUT | Data: KG-Microbe Knowledge Graph"
        )

    return interface


def main():
    """
    Main entry point for the application.
    """
    print("=" * 60)
    print(f"{config.APP_TITLE}")
    print("=" * 60)
    print()

    # Validate configuration
    print("Checking configuration...")
    config_errors = config.validate_paths()

    if config_errors:
        print("\n⚠ Configuration issues detected:")
        for error in config_errors:
            print(f"  ❌ {error}")
        print("\nPlease fix these issues in config.py before running the app.")
        print("The app will continue, but predictions will fail.\n")
    else:
        print("✓ Configuration validated successfully\n")

    # Display settings
    print("Settings:")
    print(f"  Model: {config.MODEL_PATH}")
    print(f"  Data:  {config.DATA_PATH}")
    print(f"  Type:  {config.MODEL_TYPE}")
    print(f"  Device: {config.DEFAULT_DEVICE}")
    print()

    # Create and launch interface
    print("Creating Gradio interface...")
    interface = create_interface()

    print("Launching application...")
    print("=" * 60)
    print()

    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()
