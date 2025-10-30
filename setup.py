#!/usr/bin/env python3
"""
Setup script for MicroGrowLinkService

Downloads model and data files from Google Drive and sets up the environment.
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
import json
import shutil

# Google Drive file IDs
GOOGLE_DRIVE_FILES = {
    'kgm_data.zip': '1b8d5aTlMvL-gANxHGwMZsZlI6J0jkD0-',
    'kogut_data.zip': '1rc17Xeh1JR-GPz81rc9PQqQQeYrzU7w9',
    'kogut_20251026_212314.pt': '1CQV7dVPnKHqG39zER6OKvn-x_WWevRkf'
}

# Expected file sizes (approximate, in MB)
EXPECTED_SIZES = {
    'kgm_data.zip': 420,
    'kogut_data.zip': 45,
    'kogut_20251026_212314.pt': 150
}

BASE_DIR = Path(__file__).parent
DOWNLOAD_DIR = BASE_DIR / 'downloads'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step_num, text):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 60)


def check_gdown():
    """Check if gdown is installed."""
    try:
        subprocess.run(['gdown', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_gdown():
    """Install gdown package."""
    print("Installing gdown...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown'], check=True)
        print("✓ gdown installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install gdown: {e}")
        return False


def download_file(file_id, output_path):
    """Download a file from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading {output_path.name}...")

    try:
        subprocess.run(
            ['gdown', url, '-O', str(output_path)],
            check=True,
            capture_output=False
        )

        # Check file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded {output_path.name} ({size_mb:.1f} MB)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {output_path.name}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"✓ Extracted to {extract_to}")
        return True

    except Exception as e:
        print(f"✗ Failed to extract {zip_path.name}: {e}")
        return False


def verify_installation():
    """Verify that all required files are in place."""
    print("\nVerifying installation...")

    required_files = [
        DATA_DIR / 'merged-kg_edges.tsv',
        DATA_DIR / 'merged-kg_nodes.tsv',
        DATA_DIR / 'kogut' / 'vocabularies.json',
        MODELS_DIR / 'kogut_20251026_212314.pt'
    ]

    all_present = True
    for file_path in required_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_path.relative_to(BASE_DIR)} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Missing: {file_path.relative_to(BASE_DIR)}")
            all_present = False

    return all_present


def verify_vocabularies():
    """Verify vocabularies.json has expected content."""
    vocab_path = DATA_DIR / 'kogut' / 'vocabularies.json'

    if not vocab_path.exists():
        return False

    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        entities = vocab.get('entities', vocab.get('entity2id', {}))
        relations = vocab.get('relations', vocab.get('relation2id', {}))

        print(f"\n✓ Vocabularies: {len(entities):,} entities, {len(relations)} relations")

        return len(entities) > 1000000  # Should have 1.3M+ entities

    except Exception as e:
        print(f"✗ Failed to verify vocabularies: {e}")
        return False


def generate_isolation_hierarchy():
    """Generate the isolation source hierarchy file."""
    print("\nGenerating isolation source hierarchy...")

    script_path = BASE_DIR / 'scripts' / 'build_isolation_source_hierarchy.py'

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    try:
        # Run using uv if available, otherwise python
        try:
            subprocess.run(['uv', 'run', 'python', str(script_path)], check=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, str(script_path)], check=True)

        hierarchy_path = DATA_DIR / 'isolation_source_hierarchy.json'
        if hierarchy_path.exists():
            print(f"✓ Generated {hierarchy_path.name}")
            return True
        else:
            print(f"✗ Failed to generate {hierarchy_path.name}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate hierarchy: {e}")
        return False


def cleanup_downloads():
    """Clean up downloaded zip files."""
    print("\nCleaning up downloads...")

    if DOWNLOAD_DIR.exists():
        try:
            shutil.rmtree(DOWNLOAD_DIR)
            print("✓ Cleaned up download directory")
        except Exception as e:
            print(f"⚠ Could not clean up downloads: {e}")


def main():
    """Main setup function."""
    print_header("MicroGrowLinkService Setup")
    print("This script will download and set up all required files.")
    print(f"Total download size: ~615 MB")
    print(f"Disk space required: ~3 GB")

    # Ask for confirmation
    response = input("\nContinue with setup? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Setup cancelled.")
        return 1

    # Step 1: Check/install gdown
    print_step(1, "Checking dependencies")

    if not check_gdown():
        print("gdown not found. Installing...")
        if not install_gdown():
            print("\n✗ Setup failed: Could not install gdown")
            print("Please install gdown manually: pip install gdown")
            return 1
    else:
        print("✓ gdown is installed")

    # Step 2: Create directories
    print_step(2, "Creating directories")

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"✓ Created directory structure")

    # Step 3: Download files
    print_step(3, "Downloading files from Google Drive")
    print("This may take 10-30 minutes depending on your connection...\n")

    downloads_successful = True

    for filename, file_id in GOOGLE_DRIVE_FILES.items():
        output_path = DOWNLOAD_DIR / filename

        # Skip if already downloaded
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"⚠ {filename} already exists ({size_mb:.1f} MB), skipping download")
            continue

        if not download_file(file_id, output_path):
            downloads_successful = False
            break

    if not downloads_successful:
        print("\n✗ Setup failed: Could not download all files")
        return 1

    # Step 4: Extract files
    print_step(4, "Extracting files")

    # Extract kgm_data.zip to data/
    kgm_zip = DOWNLOAD_DIR / 'kgm_data.zip'
    if kgm_zip.exists():
        if not extract_zip(kgm_zip, DATA_DIR):
            print("\n✗ Setup failed: Could not extract KG data")
            return 1

    # Extract kogut_data.zip to data/
    kogut_zip = DOWNLOAD_DIR / 'kogut_data.zip'
    if kogut_zip.exists():
        if not extract_zip(kogut_zip, DATA_DIR):
            print("\n✗ Setup failed: Could not extract KOGUT data")
            return 1

    # Move model file to models/
    model_file = DOWNLOAD_DIR / 'kogut_20251026_212314.pt'
    model_dest = MODELS_DIR / 'kogut_20251026_212314.pt'

    if model_file.exists() and not model_dest.exists():
        print(f"Moving model file to {MODELS_DIR}...")
        shutil.move(str(model_file), str(model_dest))
        print(f"✓ Model file in place")

    # Step 5: Verify installation
    print_step(5, "Verifying installation")

    if not verify_installation():
        print("\n✗ Setup failed: Some files are missing")
        return 1

    if not verify_vocabularies():
        print("\n✗ Setup failed: Vocabularies file is invalid")
        return 1

    # Step 6: Generate isolation hierarchy
    print_step(6, "Generating isolation source hierarchy")

    if not generate_isolation_hierarchy():
        print("\n⚠ Warning: Could not generate isolation source hierarchy")
        print("You can run this manually later:")
        print("  uv run python scripts/build_isolation_source_hierarchy.py")

    # Step 7: Clean up
    print_step(7, "Cleaning up")
    cleanup_downloads()

    # Success!
    print_header("Setup Complete!")

    print("✓ All files downloaded and extracted successfully")
    print(f"✓ Model: {MODELS_DIR / 'kogut_20251026_212314.pt'}")
    print(f"✓ Data: {DATA_DIR}")
    print()
    print("Next steps:")
    print("  1. Install dependencies: uv sync")
    print("  2. Run the application: uv run python app.py")
    print()
    print("The app will be available at: http://localhost:7860")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
