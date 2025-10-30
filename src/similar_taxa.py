"""
Similar taxa finder using DuckDB queries on KG-Microbe knowledge graph.
Finds taxa with similar trait profiles using Hamming distance.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import config


def calculate_hamming_distance(profile1: Dict[str, str], profile2: Dict[str, str]) -> Tuple[int, int]:
    """
    Calculate Hamming distance between two trait profiles.
    Only compares traits that are present in profile1 (query profile).

    Args:
        profile1: Query trait profile (from user input)
        profile2: Target trait profile (from KG taxa)

    Returns:
        Tuple of (mismatches, total_compared)
    """
    mismatches = 0
    total_compared = 0

    for trait_type, trait_value in profile1.items():
        if trait_type in profile2:
            total_compared += 1
            if profile2[trait_type] != trait_value:
                mismatches += 1

    return mismatches, total_compared


def find_similar_taxa(features_dict: Dict[str, str],
                      top_k: int = 10,
                      min_traits_pct: float = 100.0,
                      kg_edges_path: Path = None) -> pd.DataFrame:
    """
    Find taxa with similar trait profiles using DuckDB.

    Args:
        features_dict: Dictionary of trait type -> value from user input
        top_k: Number of similar taxa to return
        min_traits_pct: Minimum percentage of input traits that must be matched (0-100)
        kg_edges_path: Path to KG edges file

    Returns:
        DataFrame with columns: taxon, similarity, distance, traits_matched, traits_matched_pct, media
    """
    if kg_edges_path is None:
        kg_edges_path = config.MICROGROWLINK_DIR / "data" / "merged-kg_edges.tsv"

    if not kg_edges_path.exists():
        return pd.DataFrame()

    # Total traits in user query
    total_query_traits = len(features_dict)

    if total_query_traits == 0:
        return pd.DataFrame()

    # Connect to DuckDB
    con = duckdb.connect(':memory:')

    try:
        # Load the KG edges into DuckDB
        con.execute(f"""
            CREATE TABLE kg_edges AS
            SELECT * FROM read_csv_auto('{kg_edges_path}', delim='\t', header=true)
        """)

        # Get all taxa with their traits
        trait_categories = list(features_dict.keys())
        trait_prefixes = [f"{cat}:" for cat in trait_categories]

        # Build query to get taxa and their traits
        taxa_traits_query = """
        SELECT
            subject as taxon,
            object as trait
        FROM kg_edges
        WHERE predicate = 'biolink:has_phenotype'
        """

        # Add filter for relevant trait categories
        if trait_prefixes:
            prefix_conditions = " OR ".join([f"object LIKE '{prefix}%'" for prefix in trait_prefixes])
            taxa_traits_query += f" AND ({prefix_conditions})"

        taxa_traits = con.execute(taxa_traits_query).df()

        if taxa_traits.empty:
            return pd.DataFrame()

        # Group traits by taxon
        taxon_profiles = {}
        for _, row in taxa_traits.iterrows():
            taxon = row['taxon']
            trait = row['trait']

            # Parse trait into type:value
            if ':' in trait:
                trait_type, trait_value = trait.split(':', 1)
                if taxon not in taxon_profiles:
                    taxon_profiles[taxon] = {}
                taxon_profiles[taxon][trait_type] = trait_value

        # Calculate similarity for each taxon
        similarities = []
        for taxon, profile in taxon_profiles.items():
            distance, total = calculate_hamming_distance(features_dict, profile)

            if total > 0:  # Only include taxa with at least one matching trait category
                # Calculate percentage of input traits that were matched
                traits_matched_pct = (total / total_query_traits) * 100

                # Only include if meets minimum threshold
                if traits_matched_pct >= min_traits_pct:
                    similarity = 1 - (distance / total)
                    similarities.append({
                        'taxon': taxon,
                        'similarity': similarity,
                        'distance': distance,
                        'traits_matched': total,
                        'traits_matched_pct': traits_matched_pct,
                        'profile': profile
                    })

        if not similarities:
            return pd.DataFrame()

        # Sort by similarity (descending), then by traits_matched_pct (descending), and get top k
        similarities_df = pd.DataFrame(similarities)
        similarities_df = similarities_df.sort_values(
            ['similarity', 'traits_matched_pct'],
            ascending=[False, False]
        ).head(top_k)

        # Get media for these taxa
        taxa_list = similarities_df['taxon'].tolist()
        taxa_str = "', '".join(taxa_list)

        media_query = f"""
        SELECT
            subject as taxon,
            object as medium
        FROM kg_edges
        WHERE predicate = 'biolink:occurs_in'
        AND subject IN ('{taxa_str}')
        AND object LIKE 'medium:%'
        """

        taxa_media = con.execute(media_query).df()

        # Group media by taxon
        taxon_media = taxa_media.groupby('taxon')['medium'].apply(list).to_dict()

        # Add media to results
        similarities_df['media'] = similarities_df['taxon'].map(
            lambda x: taxon_media.get(x, [])
        )

        # Format trait profile for display
        similarities_df['trait_profile'] = similarities_df['profile'].apply(
            lambda p: ', '.join([f"{k}:{v}" for k, v in p.items()])
        )

        # Drop the profile dict column
        similarities_df = similarities_df.drop('profile', axis=1)

        return similarities_df

    finally:
        con.close()


def get_labels(entity_ids: List[str], kg_nodes_path: Path = None) -> Dict[str, str]:
    """
    Get labels for entity IDs from the KG nodes table.

    Args:
        entity_ids: List of entity IDs to get labels for
        kg_nodes_path: Path to KG nodes file

    Returns:
        Dictionary mapping entity_id -> label
    """
    if not entity_ids:
        return {}

    if kg_nodes_path is None:
        kg_nodes_path = config.MICROGROWLINK_DIR / "data" / "merged-kg_nodes.tsv"

    if not kg_nodes_path.exists():
        return {}

    con = duckdb.connect(':memory:')

    try:
        # Load the KG nodes
        con.execute(f"""
            CREATE TABLE kg_nodes AS
            SELECT * FROM read_csv_auto('{kg_nodes_path}', delim='\t', header=true)
        """)

        # Get labels for these entities
        entity_str = "', '".join(entity_ids)
        labels_query = f"""
        SELECT
            id as entity_id,
            name as label
        FROM kg_nodes
        WHERE id IN ('{entity_str}')
        """

        labels_df = con.execute(labels_query).df()

        # Convert to dictionary
        labels_dict = dict(zip(labels_df['entity_id'], labels_df['label']))

        return labels_dict

    finally:
        con.close()


def format_similar_taxa_for_display(similar_taxa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format similar taxa results for Gradio display.

    Args:
        similar_taxa_df: Raw similar taxa DataFrame

    Returns:
        Formatted DataFrame for display
    """
    if similar_taxa_df.empty:
        return pd.DataFrame(columns=[
            'Taxon', 'Taxon Label', 'Isolation Source', 'Traits Matched', 'Traits Matched %',
            'Trait Profile', 'Media Count', 'Media (sample)'
        ])

    display_df = similar_taxa_df.copy()

    # Get all unique entity IDs (taxa + media)
    all_taxa = display_df['taxon'].unique().tolist()
    all_media = []
    for media_list in display_df['media']:
        all_media.extend(media_list)
    all_media = list(set(all_media))

    # Fetch labels
    all_entities = all_taxa + all_media
    labels = get_labels(all_entities)

    # Extract isolation source from profile if present
    def extract_isolation_source(profile):
        if 'isolation_source' in profile:
            return profile['isolation_source']
        return ''

    # Format columns
    display_df['Taxon'] = display_df['taxon']
    display_df['Taxon Label'] = display_df['taxon'].map(lambda x: labels.get(x, ''))
    display_df['Isolation Source'] = display_df['profile'].apply(extract_isolation_source)
    display_df['Traits Matched'] = display_df['traits_matched'].astype(int)
    display_df['Traits Matched %'] = display_df['traits_matched_pct'].round(1).astype(str) + '%'
    display_df['Trait Profile'] = display_df['trait_profile']
    display_df['Media Count'] = display_df['media'].apply(len)
    display_df['Media (sample)'] = display_df['media'].apply(
        lambda media_list: ', '.join([f"{m} ({labels.get(m, '')})" for m in media_list[:5]]) if len(media_list) > 0 else 'None'
    )

    # Select columns for display
    display_columns = [
        'Taxon', 'Taxon Label', 'Isolation Source', 'Traits Matched', 'Traits Matched %',
        'Trait Profile', 'Media Count', 'Media (sample)'
    ]

    return display_df[display_columns]
