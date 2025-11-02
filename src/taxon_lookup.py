"""
Taxon trait lookup using DuckDB queries on KG-Microbe knowledge graph.
Retrieves traits for NCBITaxon IDs or BacDive strain IDs.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import config


def find_isolation_source_theme(isolation_source: str) -> Optional[str]:
    """
    Find the theme for a given isolation source value.

    Args:
        isolation_source: Specific isolation source value

    Returns:
        Theme name if found, None otherwise
    """
    for theme, sources in config.ISOLATION_SOURCE_HIERARCHY.items():
        for source_item in sources:
            if source_item.get('value') == isolation_source:
                return theme
    return None


def lookup_taxon_traits(taxon_id: str,
                       kg_edges_path: Path = None,
                       kg_nodes_path: Path = None) -> Tuple[Dict[str, str], str, str]:
    """
    Lookup traits for a given taxon ID from the knowledge graph.

    Args:
        taxon_id: NCBITaxon ID (e.g., "NCBITaxon:372072") or BacDive strain ID
        kg_edges_path: Path to KG edges file
        kg_nodes_path: Path to KG nodes file

    Returns:
        Tuple of (traits_dict, taxon_label, error_message)
        - traits_dict: Dictionary mapping trait type to value (e.g., {"temp_opt": "mid2"})
        - taxon_label: Human-readable name of the taxon
        - error_message: Error message if lookup failed, empty string otherwise
    """
    if kg_edges_path is None:
        kg_edges_path = config.MICROGROWLINK_DIR / "data" / "merged-kg_edges.tsv"

    if kg_nodes_path is None:
        kg_nodes_path = config.MICROGROWLINK_DIR / "data" / "merged-kg_nodes.tsv"

    if not kg_edges_path.exists():
        return {}, "", f"KG edges file not found: {kg_edges_path}"

    if not kg_nodes_path.exists():
        return {}, "", f"KG nodes file not found: {kg_nodes_path}"

    # Normalize taxon ID format
    taxon_id = taxon_id.strip()

    # If user didn't include prefix, try to infer it
    if not taxon_id.startswith("NCBITaxon:") and not taxon_id.startswith("bacdive.taxon:"):
        # If it's just a number, assume NCBITaxon
        if taxon_id.isdigit():
            taxon_id = f"NCBITaxon:{taxon_id}"
        # If it starts with a letter, might be BacDive
        elif taxon_id[0].isalpha():
            taxon_id = f"bacdive.taxon:{taxon_id}"

    con = duckdb.connect(':memory:')

    try:
        # Load the KG edges
        con.execute(f"""
            CREATE TABLE kg_edges AS
            SELECT * FROM read_csv_auto('{kg_edges_path}', delim='\t', header=true)
        """)

        # Load the KG nodes
        con.execute(f"""
            CREATE TABLE kg_nodes AS
            SELECT * FROM read_csv_auto('{kg_nodes_path}', delim='\t', header=true)
        """)

        # Check if taxon exists
        taxon_check_query = f"""
        SELECT id, name, category
        FROM kg_nodes
        WHERE id = '{taxon_id}'
        """

        taxon_info = con.execute(taxon_check_query).df()

        if taxon_info.empty:
            return {}, "", f"Taxon ID '{taxon_id}' not found in knowledge graph. Please check the ID and try again."

        taxon_label = taxon_info.iloc[0]['name'] if not pd.isna(taxon_info.iloc[0]['name']) else taxon_id
        taxon_category = taxon_info.iloc[0]['category'] if not pd.isna(taxon_info.iloc[0]['category']) else ""

        # Get all traits for this taxon
        traits_query = f"""
        SELECT
            object as trait
        FROM kg_edges
        WHERE subject = '{taxon_id}'
        AND predicate = 'biolink:has_phenotype'
        """

        traits_df = con.execute(traits_query).df()

        if traits_df.empty:
            return {}, taxon_label, f"No traits found for taxon '{taxon_label}' ({taxon_id}) in the knowledge graph."

        # Parse traits into dictionary
        traits_dict = {}
        for _, row in traits_df.iterrows():
            trait = row['trait']
            if ':' in trait:
                trait_type, trait_value = trait.split(':', 1)
                # Only keep traits that match our feature categories
                if trait_type in config.FEATURE_CATEGORIES.keys():
                    traits_dict[trait_type] = trait_value

        if not traits_dict:
            return {}, taxon_label, f"Found {len(traits_df)} traits for '{taxon_label}', but none match the expected trait categories used by the model."

        return traits_dict, taxon_label, ""

    except Exception as e:
        return {}, "", f"Error querying knowledge graph: {str(e)}"

    finally:
        con.close()


def get_taxon_media(taxon_id: str, kg_edges_path: Path = None) -> list:
    """
    Get all media associated with a taxon.

    Args:
        taxon_id: Taxon ID
        kg_edges_path: Path to KG edges file

    Returns:
        List of medium IDs
    """
    if kg_edges_path is None:
        kg_edges_path = config.MICROGROWLINK_DIR / "data" / "merged-kg_edges.tsv"

    if not kg_edges_path.exists():
        return []

    con = duckdb.connect(':memory:')

    try:
        # Load the KG edges
        con.execute(f"""
            CREATE TABLE kg_edges AS
            SELECT * FROM read_csv_auto('{kg_edges_path}', delim='\t', header=true)
        """)

        # Get media for this taxon
        media_query = f"""
        SELECT
            object as medium
        FROM kg_edges
        WHERE subject = '{taxon_id}'
        AND predicate = 'biolink:occurs_in'
        AND object LIKE 'medium:%'
        """

        media_df = con.execute(media_query).df()

        return media_df['medium'].tolist()

    except Exception as e:
        print(f"Error getting media for taxon: {str(e)}")
        return []

    finally:
        con.close()
