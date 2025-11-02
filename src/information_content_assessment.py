#!/usr/bin/env python3
"""
Information Content Assessment for Trait Profiles

Evaluates the information content and predictive power of user-provided trait
profiles using information-theoretic metrics, statistical significance testing,
and feature importance scoring.

This module provides advanced assessment beyond basic validation, measuring:
- Shannon entropy and mutual information
- Statistical significance of feature-media associations
- Bootstrap confidence intervals
- Per-feature predictive power and importance

Author: MicroGrowLink Team
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InformationContentMetrics:
    """
    Comprehensive metrics for assessing information content of a trait profile.

    Information-Theoretic Metrics:
        feature_entropy: Shannon entropy of feature distribution H(Features)
        media_entropy: Shannon entropy of media distribution H(Media)
        joint_entropy: Joint entropy H(Features, Media)
        mutual_information: Information shared between features and media I(F;M)
        conditional_entropy: Uncertainty about media given features H(M|F)
        information_gain: Reduction in uncertainty about media
        normalized_mutual_info: Mutual information normalized by entropy

    Statistical Significance:
        fisher_pvalues: P-values from Fisher's exact test per feature
        overall_significance: Combined p-value across all features
        bootstrap_ci_lower: Lower bound of 95% bootstrap confidence interval
        bootstrap_ci_upper: Upper bound of 95% bootstrap confidence interval
        significant_features: Number of statistically significant features

    Feature Importance:
        feature_importance: Ranked importance scores per feature (0-1)
        predictive_power: Predictive strength of each feature (0-1)
        information_gain_per_feature: Information gain contributed by each feature
        feature_ranking: Features sorted by importance (best to worst)

    Assessment Summary:
        information_sufficiency: Overall sufficiency score (0-1)
        confidence_level: Assessment confidence ('high', 'medium', 'low')
        assessment_passed: Whether profile has sufficient information
        num_matching_taxa: Number of similar taxa found in KG
        warnings: List of warning messages
        recommendations: List of suggestions to improve profile
    """
    # Information Theory
    feature_entropy: float = 0.0
    media_entropy: float = 0.0
    joint_entropy: float = 0.0
    mutual_information: float = 0.0
    conditional_entropy: float = 0.0
    information_gain: float = 0.0
    normalized_mutual_info: float = 0.0

    # Statistical Significance
    fisher_pvalues: Dict[str, float] = field(default_factory=dict)
    overall_significance: float = 1.0
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    significant_features: int = 0

    # Feature Importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    predictive_power: Dict[str, float] = field(default_factory=dict)
    information_gain_per_feature: Dict[str, float] = field(default_factory=dict)
    feature_ranking: List[Tuple[str, float]] = field(default_factory=list)

    # Summary
    information_sufficiency: float = 0.0
    confidence_level: str = "unknown"
    assessment_passed: bool = False
    num_matching_taxa: int = 0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'information_theory': {
                'feature_entropy': self.feature_entropy,
                'media_entropy': self.media_entropy,
                'joint_entropy': self.joint_entropy,
                'mutual_information': self.mutual_information,
                'conditional_entropy': self.conditional_entropy,
                'information_gain': self.information_gain,
                'normalized_mutual_info': self.normalized_mutual_info,
            },
            'statistical_significance': {
                'fisher_pvalues': self.fisher_pvalues,
                'overall_significance': self.overall_significance,
                'bootstrap_ci': {
                    'lower': self.bootstrap_ci_lower,
                    'upper': self.bootstrap_ci_upper,
                },
                'significant_features': self.significant_features,
            },
            'feature_importance': {
                'importance_scores': self.feature_importance,
                'predictive_power': self.predictive_power,
                'information_gain': self.information_gain_per_feature,
                'ranking': self.feature_ranking,
            },
            'summary': {
                'information_sufficiency': self.information_sufficiency,
                'confidence_level': self.confidence_level,
                'assessment_passed': self.assessment_passed,
                'num_matching_taxa': self.num_matching_taxa,
                'warnings': self.warnings,
                'recommendations': self.recommendations,
            }
        }


class InformationContentAssessor:
    """
    Assess the information content of trait profiles for media prediction.

    Uses information theory, statistical testing, and feature importance
    analysis to evaluate whether a trait profile contains sufficient
    information for robust media predictions.
    """

    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        min_matching_taxa: int = 10,
        significance_level: float = 0.05,
        bootstrap_iterations: int = 1000,
        random_seed: int = 42
    ):
        """
        Initialize the information content assessor.

        Args:
            nodes_file: Path to KG nodes TSV file
            edges_file: Path to KG edges TSV file
            min_matching_taxa: Minimum number of matching taxa required
            significance_level: Alpha level for statistical tests (default: 0.05)
            bootstrap_iterations: Number of bootstrap samples (default: 1000)
            random_seed: Random seed for reproducibility
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.min_matching_taxa = min_matching_taxa
        self.significance_level = significance_level
        self.bootstrap_iterations = bootstrap_iterations
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)

        # Load KG data
        logger.info(f"Loading KG data from {nodes_file} and {edges_file}")
        self._load_kg_data()

        logger.info(f"Loaded {len(self.taxa_features)} taxa with features")
        logger.info(f"Loaded {len(self.taxa_media)} taxa-media relationships")

    def _load_kg_data(self):
        """Load and prepare knowledge graph data."""
        # Load nodes
        try:
            nodes_df = pd.read_csv(self.nodes_file, sep='\t', on_bad_lines='skip')
            logger.info(f"Loaded {len(nodes_df)} nodes")
        except Exception as e:
            logger.error(f"Failed to load nodes file: {e}")
            raise

        # Load edges
        try:
            edges_df = pd.read_csv(self.edges_file, sep='\t', on_bad_lines='skip')
            logger.info(f"Loaded {len(edges_df)} edges")
        except Exception as e:
            logger.error(f"Failed to load edges file: {e}")
            raise

        # Extract taxa entities
        taxa_ids = set(nodes_df[nodes_df['id'].str.startswith('NCBITaxon:') |
                                 nodes_df['id'].str.startswith('strain:')]['id'].values)
        logger.info(f"Found {len(taxa_ids)} taxa entities")

        # Extract feature relationships (has_phenotype, capable_of, etc.)
        feature_predicates = ['biolink:has_phenotype', 'biolink:capable_of']
        feature_edges = edges_df[
            edges_df['predicate'].isin(feature_predicates) &
            edges_df['subject'].isin(taxa_ids)
        ]

        # Build taxa -> features mapping
        self.taxa_features = defaultdict(set)
        for _, row in feature_edges.iterrows():
            self.taxa_features[row['subject']].add(row['object'])

        # Extract media relationships
        media_edges = edges_df[
            (edges_df['predicate'] == 'biolink:occurs_in') &
            edges_df['subject'].isin(taxa_ids)
        ]

        # Build taxa -> media mapping
        self.taxa_media = defaultdict(set)
        for _, row in media_edges.iterrows():
            self.taxa_media[row['subject']].add(row['object'])

        # Build reverse indices for fast lookup
        self.feature_to_taxa = defaultdict(set)
        for taxon, features in self.taxa_features.items():
            for feature in features:
                self.feature_to_taxa[feature].add(taxon)

        self.media_to_taxa = defaultdict(set)
        for taxon, media_list in self.taxa_media.items():
            for medium in media_list:
                self.media_to_taxa[medium].add(taxon)

    def _find_matching_taxa(
        self,
        feature_ids: List[str],
        min_overlap: int = 2
    ) -> List[str]:
        """
        Find taxa that share features with the query profile.

        Args:
            feature_ids: List of feature IDs in query profile
            min_overlap: Minimum number of shared features

        Returns:
            List of matching taxon IDs
        """
        if not feature_ids:
            return []

        # Find taxa with each feature
        taxa_sets = [self.feature_to_taxa.get(feat, set()) for feat in feature_ids]

        # Count overlaps
        taxon_counts = Counter()
        for taxa_set in taxa_sets:
            for taxon in taxa_set:
                taxon_counts[taxon] += 1

        # Filter by minimum overlap
        matching_taxa = [
            taxon for taxon, count in taxon_counts.items()
            if count >= min(min_overlap, len(feature_ids))
        ]

        return matching_taxa

    def compute_shannon_entropy(self, values: List) -> float:
        """
        Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x)

        Args:
            values: List of observed values

        Returns:
            Shannon entropy in bits
        """
        if not values:
            return 0.0

        # Count frequencies
        counts = Counter(values)
        total = len(values)

        # Compute probabilities and entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def compute_mutual_information(
        self,
        features_list: List[Tuple],
        media_list: List[str]
    ) -> float:
        """
        Compute mutual information I(Features; Media)

        I(X;Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)

        Args:
            features_list: List of feature tuples for each taxon
            media_list: List of media for each taxon

        Returns:
            Mutual information in bits
        """
        if not features_list or not media_list:
            return 0.0

        # Compute individual entropies
        h_features = self.compute_shannon_entropy(features_list)
        h_media = self.compute_shannon_entropy(media_list)

        # Compute joint entropy
        joint_values = list(zip(features_list, media_list))
        h_joint = self.compute_shannon_entropy(joint_values)

        # MI = H(X) + H(Y) - H(X,Y)
        mutual_info = h_features + h_media - h_joint

        return max(0.0, mutual_info)  # Ensure non-negative

    def test_feature_significance(
        self,
        feature_id: str,
        matching_taxa: List[str]
    ) -> float:
        """
        Test statistical significance of feature-media association using Fisher's exact test.

        Args:
            feature_id: Feature to test
            matching_taxa: List of relevant taxa

        Returns:
            P-value from Fisher's exact test
        """
        # Build 2x2 contingency table
        # Has feature × Grows in specific media

        # Get all media from matching taxa
        all_media = set()
        for taxon in matching_taxa:
            all_media.update(self.taxa_media.get(taxon, set()))

        if not all_media:
            return 1.0

        # Get taxa with this feature
        taxa_with_feature = self.feature_to_taxa.get(feature_id, set())

        # For each medium, compute enrichment
        p_values = []
        for medium in all_media:
            taxa_with_medium = self.media_to_taxa.get(medium, set())

            # 2x2 table
            # a: has feature AND has medium
            a = len(taxa_with_feature & taxa_with_medium & set(matching_taxa))
            # b: has feature AND NOT has medium
            b = len(taxa_with_feature & (set(matching_taxa) - taxa_with_medium))
            # c: NOT has feature AND has medium
            c = len((set(matching_taxa) - taxa_with_feature) & taxa_with_medium)
            # d: NOT has feature AND NOT has medium
            d = len(set(matching_taxa) - taxa_with_feature - taxa_with_medium)

            # Fisher's exact test
            if a + b > 0 and c + d > 0 and a + c > 0:
                try:
                    _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                    p_values.append(p_value)
                except:
                    pass

        # Return minimum p-value (strongest association)
        return min(p_values) if p_values else 1.0

    def compute_bootstrap_ci(
        self,
        matching_taxa: List[str],
        n_iterations: int = None
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for media predictions.

        Args:
            matching_taxa: List of matching taxon IDs
            n_iterations: Number of bootstrap samples (default: use class setting)

        Returns:
            Tuple of (lower_bound, upper_bound) for 95% CI
        """
        if n_iterations is None:
            n_iterations = self.bootstrap_iterations

        if not matching_taxa:
            return (0.0, 0.0)

        # Get media counts for each taxon
        media_counts = [len(self.taxa_media.get(taxon, set())) for taxon in matching_taxa]

        if not media_counts:
            return (0.0, 0.0)

        # Bootstrap resampling
        bootstrap_means = []
        n_taxa = len(media_counts)

        for _ in range(n_iterations):
            # Resample with replacement
            sample = np.random.choice(media_counts, size=n_taxa, replace=True)
            bootstrap_means.append(np.mean(sample))

        # Compute 95% CI
        lower = np.percentile(bootstrap_means, 2.5)
        upper = np.percentile(bootstrap_means, 97.5)

        return (lower, upper)

    def compute_feature_importance(
        self,
        feature_ids: List[str],
        matching_taxa: List[str]
    ) -> Dict[str, float]:
        """
        Compute importance score for each feature based on predictive power.

        Uses information gain: IG(Media|Feature) = H(Media) - H(Media|Feature)

        Args:
            feature_ids: List of features in profile
            matching_taxa: List of matching taxa

        Returns:
            Dictionary mapping feature_id -> importance score (0-1)
        """
        importance_scores = {}

        # Get baseline media entropy
        all_media = []
        for taxon in matching_taxa:
            all_media.extend(self.taxa_media.get(taxon, set()))

        baseline_entropy = self.compute_shannon_entropy(all_media)

        if baseline_entropy == 0:
            return {feat: 0.0 for feat in feature_ids}

        # Compute information gain for each feature
        for feature in feature_ids:
            # Split taxa by feature presence
            taxa_with_feature = [t for t in matching_taxa if feature in self.taxa_features.get(t, set())]
            taxa_without_feature = [t for t in matching_taxa if feature not in self.taxa_features.get(t, set())]

            # Compute conditional entropy H(Media|Feature)
            media_with = []
            media_without = []

            for taxon in taxa_with_feature:
                media_with.extend(self.taxa_media.get(taxon, set()))

            for taxon in taxa_without_feature:
                media_without.extend(self.taxa_media.get(taxon, set()))

            h_media_with = self.compute_shannon_entropy(media_with)
            h_media_without = self.compute_shannon_entropy(media_without)

            # Weighted conditional entropy
            p_with = len(taxa_with_feature) / len(matching_taxa) if matching_taxa else 0
            p_without = 1 - p_with

            conditional_entropy = p_with * h_media_with + p_without * h_media_without

            # Information gain
            info_gain = baseline_entropy - conditional_entropy

            # Normalize to 0-1 range
            importance = info_gain / baseline_entropy if baseline_entropy > 0 else 0.0
            importance_scores[feature] = max(0.0, min(1.0, importance))

        return importance_scores

    def assess_profile(
        self,
        features_dict: Dict[str, str],
        min_overlap: int = 2
    ) -> InformationContentMetrics:
        """
        Perform comprehensive information content assessment of trait profile.

        Args:
            features_dict: Dictionary mapping feature type to value
            min_overlap: Minimum feature overlap for matching taxa

        Returns:
            InformationContentMetrics object with all computed metrics
        """
        logger.info(f"Assessing information content for {len(features_dict)} features")

        metrics = InformationContentMetrics()

        # Convert features_dict to feature IDs
        feature_ids = [f"{k}:{v}" for k, v in features_dict.items()]

        # Find matching taxa
        matching_taxa = self._find_matching_taxa(feature_ids, min_overlap)
        metrics.num_matching_taxa = len(matching_taxa)

        logger.info(f"Found {len(matching_taxa)} matching taxa")

        if len(matching_taxa) < self.min_matching_taxa:
            metrics.warnings.append(
                f"Only {len(matching_taxa)} matching taxa found (minimum: {self.min_matching_taxa})"
            )
            metrics.recommendations.append(
                "Add more specific features to improve matching"
            )
            metrics.assessment_passed = False
            metrics.confidence_level = "low"
            return metrics

        # Compute information-theoretic metrics
        logger.info("Computing information-theoretic metrics...")

        # Get features and media for matching taxa
        taxa_features_tuples = []
        taxa_media_list = []

        for taxon in matching_taxa:
            taxon_features = tuple(sorted(self.taxa_features.get(taxon, set())))
            taxon_media = list(self.taxa_media.get(taxon, set()))

            if taxon_features and taxon_media:
                for medium in taxon_media:
                    taxa_features_tuples.append(taxon_features)
                    taxa_media_list.append(medium)

        # Shannon entropies
        metrics.feature_entropy = self.compute_shannon_entropy(taxa_features_tuples)
        metrics.media_entropy = self.compute_shannon_entropy(taxa_media_list)

        # Mutual information
        metrics.mutual_information = self.compute_mutual_information(
            taxa_features_tuples,
            taxa_media_list
        )

        # Joint and conditional entropy
        joint_values = list(zip(taxa_features_tuples, taxa_media_list))
        metrics.joint_entropy = self.compute_shannon_entropy(joint_values)
        metrics.conditional_entropy = metrics.joint_entropy - metrics.feature_entropy
        metrics.information_gain = metrics.media_entropy - metrics.conditional_entropy

        # Normalized mutual information
        if metrics.media_entropy > 0:
            metrics.normalized_mutual_info = metrics.mutual_information / metrics.media_entropy

        # Statistical significance testing
        logger.info("Testing statistical significance...")

        for feature_id in feature_ids:
            p_value = self.test_feature_significance(feature_id, matching_taxa)
            metrics.fisher_pvalues[feature_id] = p_value

            if p_value < self.significance_level:
                metrics.significant_features += 1

        # Combined p-value (Fisher's method)
        if metrics.fisher_pvalues:
            chi_squared = -2 * sum(np.log(max(p, 1e-10)) for p in metrics.fisher_pvalues.values())
            df = 2 * len(metrics.fisher_pvalues)
            metrics.overall_significance = 1 - stats.chi2.cdf(chi_squared, df)

        # Bootstrap confidence intervals
        logger.info("Computing bootstrap confidence intervals...")
        metrics.bootstrap_ci_lower, metrics.bootstrap_ci_upper = self.compute_bootstrap_ci(matching_taxa)

        # Feature importance
        logger.info("Computing feature importance scores...")
        metrics.feature_importance = self.compute_feature_importance(feature_ids, matching_taxa)

        # Predictive power (similar to importance, but based on effect size)
        metrics.predictive_power = metrics.feature_importance.copy()

        # Information gain per feature
        metrics.information_gain_per_feature = metrics.feature_importance.copy()

        # Rank features
        metrics.feature_ranking = sorted(
            metrics.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Compute overall information sufficiency score
        # Weighted combination of multiple factors
        mi_score = min(1.0, metrics.mutual_information / 2.0)  # Normalize by 2.0 bits
        sig_score = metrics.significant_features / len(feature_ids) if feature_ids else 0.0
        importance_score = np.mean(list(metrics.feature_importance.values())) if metrics.feature_importance else 0.0

        metrics.information_sufficiency = (
            0.4 * mi_score +
            0.3 * sig_score +
            0.3 * importance_score
        )

        # Determine confidence level
        if metrics.information_sufficiency >= 0.8 and metrics.mutual_information >= 2.0:
            metrics.confidence_level = "high"
            metrics.assessment_passed = True
        elif metrics.information_sufficiency >= 0.5 and metrics.mutual_information >= 1.0:
            metrics.confidence_level = "medium"
            metrics.assessment_passed = True
        else:
            metrics.confidence_level = "low"
            metrics.assessment_passed = False

        # Generate warnings and recommendations
        if metrics.mutual_information < 1.0:
            metrics.warnings.append(
                f"Low mutual information ({metrics.mutual_information:.2f} bits) - predictions may be unreliable"
            )
            metrics.recommendations.append(
                "Add more discriminative features to increase information content"
            )

        if metrics.conditional_entropy > 3.0:
            metrics.warnings.append(
                f"High conditional entropy ({metrics.conditional_entropy:.2f} bits) - high prediction uncertainty"
            )

        if metrics.significant_features < len(feature_ids) * 0.5:
            metrics.warnings.append(
                f"Only {metrics.significant_features}/{len(feature_ids)} features are statistically significant"
            )
            metrics.recommendations.append(
                "Consider replacing non-significant features with more predictive ones"
            )

        # Recommend features to add based on importance in similar profiles
        low_importance_features = [
            feat for feat, score in metrics.feature_ranking
            if score < 0.3
        ]

        if low_importance_features:
            metrics.recommendations.append(
                f"Low-importance features: {', '.join(low_importance_features)}"
            )

        logger.info(f"Assessment complete: {metrics.confidence_level} confidence")

        return metrics
