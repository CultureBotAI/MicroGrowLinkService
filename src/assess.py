"""
Information content assessment for trait profiles.
Wrapper around MicroGrowLink's InformationContentAssessor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

# Add MicroGrowLink to Python path
MICROGROWLINK_DIR = Path(__file__).parent.parent.parent / "MicroGrowLink"
sys.path.insert(0, str(MICROGROWLINK_DIR))

try:
    from src.utils.information_content_assessment import (
        InformationContentAssessor,
        InformationContentMetrics
    )
    ASSESSMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import InformationContentAssessor: {e}")
    ASSESSMENT_AVAILABLE = False
    # Define dummy types for type hints when import fails
    InformationContentMetrics = None


class TraitAssessor:
    """Wrapper for assessing information content of trait profiles."""

    def __init__(self, nodes_file: Path, edges_file: Path,
                 min_matching_taxa: int = 10,
                 significance_level: float = 0.05,
                 bootstrap_iterations: int = 1000):
        """
        Initialize the trait assessor.

        Args:
            nodes_file: Path to KG nodes TSV file
            edges_file: Path to KG edges TSV file
            min_matching_taxa: Minimum matching taxa for reliable statistics
            significance_level: Alpha level for statistical tests (default: 0.05)
            bootstrap_iterations: Number of bootstrap samples (default: 1000)
        """
        if not ASSESSMENT_AVAILABLE:
            raise ImportError("InformationContentAssessor not available. Check MicroGrowLink installation.")

        self.nodes_file = nodes_file
        self.edges_file = edges_file

        # Validate files exist
        if not nodes_file.exists():
            raise FileNotFoundError(f"KG nodes file not found: {nodes_file}")
        if not edges_file.exists():
            raise FileNotFoundError(f"KG edges file not found: {edges_file}")

        # Initialize assessor
        self.assessor = InformationContentAssessor(
            nodes_file=str(nodes_file),
            edges_file=str(edges_file),
            min_matching_taxa=min_matching_taxa,
            significance_level=significance_level,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=42  # For reproducibility
        )

    def assess(self, features_dict: Dict[str, str],
               min_overlap: int = 2) -> InformationContentMetrics:
        """
        Assess information content of a trait profile.

        Args:
            features_dict: Dictionary of feature type -> value
                          e.g., {"temp_opt": "mesophilic", "oxygen": "aerobe"}
            min_overlap: Minimum feature overlap for matching taxa

        Returns:
            InformationContentMetrics object with assessment results
        """
        return self.assessor.assess_profile(features_dict, min_overlap=min_overlap)

    def format_summary(self, metrics: InformationContentMetrics) -> str:
        """
        Format assessment metrics into a summary card (HTML).

        Args:
            metrics: Assessment metrics

        Returns:
            HTML string for summary display
        """
        # Determine confidence color
        confidence_colors = {
            "high": "#22c55e",    # Green
            "medium": "#f59e0b",  # Orange
            "low": "#ef4444"      # Red
        }
        color = confidence_colors.get(metrics.confidence_level.lower(), "#6b7280")

        # Confidence emoji
        confidence_emoji = {
            "high": "✓✓",
            "medium": "✓",
            "low": "⚠"
        }
        emoji = confidence_emoji.get(metrics.confidence_level.lower(), "?")

        summary_html = f"""
<div style="border: 2px solid {color}; border-radius: 8px; padding: 16px; margin: 10px 0; background-color: #f9fafb;">
    <h3 style="margin-top: 0; color: {color};">
        {emoji} Assessment Results - {metrics.confidence_level.upper()} Confidence
    </h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
        <div>
            <strong>Information Sufficiency:</strong><br/>
            <span style="font-size: 1.5em; color: {color};">{metrics.information_sufficiency:.2f}</span> / 1.00
        </div>
        <div>
            <strong>Mutual Information:</strong><br/>
            <span style="font-size: 1.5em;">{metrics.mutual_information:.2f}</span> bits
        </div>
        <div>
            <strong>Matching Taxa:</strong><br/>
            {metrics.num_matching_taxa} taxa
        </div>
        <div>
            <strong>Significant Features:</strong><br/>
            {metrics.significant_features} / {len(metrics.fisher_pvalues)}
        </div>
    </div>

    {"".join([f'<div style="margin-top: 12px; padding: 8px; background-color: #fef3c7; border-left: 4px solid #f59e0b;"><strong>⚠ Warning:</strong> {w}</div>' for w in metrics.warnings])}

    {"".join([f'<div style="margin-top: 8px; padding: 8px; background-color: #dbeafe; border-left: 4px solid #3b82f6;"><strong>💡 Recommendation:</strong> {r}</div>' for r in metrics.recommendations])}
</div>
"""
        return summary_html

    def format_detailed_report(self, metrics: InformationContentMetrics) -> pd.DataFrame:
        """
        Format detailed assessment metrics into a DataFrame for display.

        Args:
            metrics: Assessment metrics

        Returns:
            DataFrame with detailed metrics
        """
        # Create detailed metrics table
        details = []

        # Information Theory Metrics
        details.append({"Category": "Information Theory", "Metric": "Feature Entropy", "Value": f"{metrics.feature_entropy:.3f} bits"})
        details.append({"Category": "Information Theory", "Metric": "Media Entropy", "Value": f"{metrics.media_entropy:.3f} bits"})
        details.append({"Category": "Information Theory", "Metric": "Joint Entropy", "Value": f"{metrics.joint_entropy:.3f} bits"})
        details.append({"Category": "Information Theory", "Metric": "Mutual Information", "Value": f"{metrics.mutual_information:.3f} bits"})
        details.append({"Category": "Information Theory", "Metric": "Conditional Entropy", "Value": f"{metrics.conditional_entropy:.3f} bits"})
        details.append({"Category": "Information Theory", "Metric": "Information Gain", "Value": f"{metrics.information_gain:.3f} bits"})
        details.append({"Category": "Information Theory", "Metric": "Normalized MI", "Value": f"{metrics.normalized_mutual_info:.3f}"})

        # Statistical Significance
        details.append({"Category": "Statistical", "Metric": "Overall Significance", "Value": f"p = {metrics.overall_significance:.4f}"})
        details.append({"Category": "Statistical", "Metric": "Bootstrap CI (95%)", "Value": f"[{metrics.bootstrap_ci_lower:.1f}, {metrics.bootstrap_ci_upper:.1f}]"})
        details.append({"Category": "Statistical", "Metric": "Significant Features", "Value": f"{metrics.significant_features} / {len(metrics.fisher_pvalues)}"})

        # Summary
        details.append({"Category": "Summary", "Metric": "Information Sufficiency", "Value": f"{metrics.information_sufficiency:.3f}"})
        details.append({"Category": "Summary", "Metric": "Confidence Level", "Value": metrics.confidence_level.upper()})
        details.append({"Category": "Summary", "Metric": "Assessment Passed", "Value": "✓ Yes" if metrics.assessment_passed else "✗ No"})
        details.append({"Category": "Summary", "Metric": "Matching Taxa Count", "Value": str(metrics.num_matching_taxa)})

        return pd.DataFrame(details)

    def format_feature_importance(self, metrics: InformationContentMetrics) -> pd.DataFrame:
        """
        Format feature importance rankings into a DataFrame.

        Args:
            metrics: Assessment metrics

        Returns:
            DataFrame with feature importance rankings
        """
        if not metrics.feature_ranking:
            return pd.DataFrame(columns=["Rank", "Feature", "Importance Score", "P-value", "Significance"])

        importance_data = []
        for rank, (feature, importance) in enumerate(metrics.feature_ranking, 1):
            p_value = metrics.fisher_pvalues.get(feature, 1.0)
            significance = "✓✓ High" if p_value < 0.01 else ("✓ Significant" if p_value < 0.05 else "✗ Not significant")

            importance_data.append({
                "Rank": rank,
                "Feature": feature,
                "Importance Score": f"{importance:.3f}",
                "P-value": f"{p_value:.4f}",
                "Significance": significance
            })

        return pd.DataFrame(importance_data)


def assess_trait_profile_wrapper(features_dict: Dict[str, str],
                                   nodes_file: Path,
                                   edges_file: Path) -> Tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """
    Convenience wrapper for assessing trait profiles.

    Args:
        features_dict: Feature dictionary
        nodes_file: Path to KG nodes file
        edges_file: Path to KG edges file

    Returns:
        Tuple of (summary_html, detailed_df, feature_importance_df, confidence_level)
    """
    # Check if assessment module is available
    if not ASSESSMENT_AVAILABLE:
        error_html = """
<div style="border: 2px solid #f59e0b; border-radius: 8px; padding: 16px; background-color: #fffbeb;">
    <h3 style="color: #f59e0b; margin-top: 0;">⚠️ Assessment Module Not Available</h3>
    <p><strong>The Information Content Assessment feature requires additional dependencies from the MicroGrowLink repository.</strong></p>

    <h4>Required Dependencies:</h4>
    <ul>
        <li><code>torch_geometric</code> - PyTorch Geometric library</li>
        <li>Other MicroGrowLink dependencies</li>
    </ul>

    <h4>To enable assessment:</h4>
    <ol>
        <li>Ensure the MicroGrowLink repository is at: <code>../MicroGrowLink/</code></li>
        <li>Install MicroGrowLink dependencies: <code>cd ../MicroGrowLink && uv sync</code></li>
        <li>Verify the assessment module exists: <code>../MicroGrowLink/src/utils/information_content_assessment.py</code></li>
    </ol>

    <p><em>Note: You can still use the prediction feature without assessment. The assessment provides additional quality metrics but is not required for predictions.</em></p>
</div>
"""
        empty_df = pd.DataFrame()
        return error_html, empty_df, empty_df, "unavailable"

    try:
        assessor = TraitAssessor(nodes_file, edges_file)
        metrics = assessor.assess(features_dict)

        summary = assessor.format_summary(metrics)
        details = assessor.format_detailed_report(metrics)
        importance = assessor.format_feature_importance(metrics)

        return summary, details, importance, metrics.confidence_level

    except Exception as e:
        error_html = f"""
<div style="border: 2px solid #ef4444; border-radius: 8px; padding: 16px; background-color: #fef2f2;">
    <h3 style="color: #ef4444; margin-top: 0;">❌ Assessment Error</h3>
    <p>Failed to assess trait profile: {str(e)}</p>
</div>
"""
        empty_df = pd.DataFrame()
        return error_html, empty_df, empty_df, "error"
