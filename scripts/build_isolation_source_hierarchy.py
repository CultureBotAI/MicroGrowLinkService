"""
Build hierarchical structure for isolation sources.
Extracts isolation sources from KG and organizes them into themes.
"""

import json
import duckdb
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
MICROGROWLINK_DIR = BASE_DIR.parent / "MicroGrowLink"
KG_NODES_PATH = MICROGROWLINK_DIR / "data" / "merged-kg_nodes.tsv"
OUTPUT_PATH = BASE_DIR / "data" / "isolation_source_hierarchy.json"

def extract_isolation_sources():
    """Extract all isolation sources from KG nodes."""
    con = duckdb.connect(':memory:')

    try:
        # Load nodes
        con.execute(f"""
            CREATE TABLE kg_nodes AS
            SELECT * FROM read_csv_auto('{KG_NODES_PATH}', delim='\t', header=true)
        """)

        # Get isolation sources
        query = """
        SELECT
            id,
            name
        FROM kg_nodes
        WHERE category = 'biolink:EnvironmentalFeature'
        AND id LIKE 'isolation_source:%'
        ORDER BY name
        """

        results = con.execute(query).fetchall()

        # Convert to dict {id: name}
        sources = {row[0]: row[1] for row in results}

        return sources

    finally:
        con.close()


def create_themes(sources):
    """
    Create themed hierarchy based on source names.

    Themes:
    - Host-Associated
    - Environmental
    - Medical/Clinical
    - Laboratory/Engineered
    - Food/Agriculture
    - Other
    """

    themes = {
        "Host-Associated": [],
        "Environmental": [],
        "Medical/Clinical": [],
        "Laboratory/Engineered": [],
        "Food/Agriculture": [],
        "Other": []
    }

    # Keywords for categorization
    host_keywords = [
        'host', 'human', 'patient', 'mammals', 'animal', 'dog', 'cat', 'mouse',
        'rat', 'bird', 'fish', 'insect', 'worm', 'cattle', 'pig', 'sheep',
        'body', 'organ', 'tissue', 'blood', 'feces', 'urine', 'sputum',
        'gastrointestinal', 'intestine', 'oral', 'mouth', 'skin', 'nail',
        'respiratory', 'digestive', 'urinary', 'reproductive'
    ]

    environmental_keywords = [
        'environmental', 'aquatic', 'marine', 'freshwater', 'terrestrial',
        'soil', 'sediment', 'water', 'ocean', 'sea', 'lake', 'river',
        'forest', 'desert', 'tundra', 'arctic', 'tropical',
        'wetland', 'mangrove', 'estuary', 'coastal', 'beach', 'tidal',
        'rock', 'mineral', 'geothermal', 'hydrothermal', 'volcanic',
        'air', 'atmosphere', 'aerosol'
    ]

    medical_keywords = [
        'infection', 'disease', 'clinic', 'hospital', 'medical',
        'patient', 'wound', 'abscess', 'inflammation', 'sepsis',
        'pneumonia', 'meningitis', 'endocarditis', 'bacteremia',
        'pathogen', 'nosocomial', 'cystic-fibrosis'
    ]

    lab_keywords = [
        'laboratory', 'engineered', 'culture', 'bioreactor',
        'fermentation', 'industrial', 'biotechnology'
    ]

    food_keywords = [
        'food', 'dairy', 'milk', 'cheese', 'yogurt', 'meat',
        'vegetable', 'fruit', 'grain', 'fermented', 'beverage',
        'wine', 'beer', 'bread', 'agricultural', 'farm', 'crop',
        'plant', 'leaf', 'root', 'seed', 'compost'
    ]

    for source_id, name in sources.items():
        # Extract value part (after 'isolation_source:')
        value = source_id.replace('isolation_source:', '')
        name_lower = name.lower()
        value_lower = value.lower()

        # Check which theme matches
        categorized = False

        # Check host-associated
        if any(kw in name_lower or kw in value_lower for kw in host_keywords):
            themes["Host-Associated"].append({"id": source_id, "value": value, "label": name})
            categorized = True
        # Check medical
        elif any(kw in name_lower or kw in value_lower for kw in medical_keywords):
            themes["Medical/Clinical"].append({"id": source_id, "value": value, "label": name})
            categorized = True
        # Check environmental
        elif any(kw in name_lower or kw in value_lower for kw in environmental_keywords):
            themes["Environmental"].append({"id": source_id, "value": value, "label": name})
            categorized = True
        # Check food
        elif any(kw in name_lower or kw in value_lower for kw in food_keywords):
            themes["Food/Agriculture"].append({"id": source_id, "value": value, "label": name})
            categorized = True
        # Check lab
        elif any(kw in name_lower or kw in value_lower for kw in lab_keywords):
            themes["Laboratory/Engineered"].append({"id": source_id, "value": value, "label": name})
            categorized = True

        # If not categorized, add to Other
        if not categorized:
            themes["Other"].append({"id": source_id, "value": value, "label": name})

    # Sort each theme by label
    for theme in themes:
        themes[theme] = sorted(themes[theme], key=lambda x: x['label'])

    return themes


def main():
    print("Extracting isolation sources from KG...")
    sources = extract_isolation_sources()
    print(f"Found {len(sources)} isolation sources")

    print("\nCreating themed hierarchy...")
    themes = create_themes(sources)

    # Print summary
    for theme, items in themes.items():
        print(f"{theme}: {len(items)} items")

    # Save to JSON
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(themes, f, indent=2)

    print(f"\nSaved hierarchy to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
