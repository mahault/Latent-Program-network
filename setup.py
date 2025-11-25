"""
Master Setup Script
Downloads and generates all necessary data for LPN experiments
"""

import subprocess
import sys
from pathlib import Path
import json


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required = ['torch', 'numpy', 'matplotlib', 'seaborn', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} - MISSING")
    
    if missing:
        print(f"\nInstalling missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✓ Dependencies installed")
    else:
        print("✓ All dependencies present")


def generate_list_data():
    """Generate list operations dataset"""
    print("\n" + "="*60)
    print("Generating List Operations Dataset")
    print("="*60)
    
    if Path('data/list_ops_data').exists():
        print("✓ list_ops_data already exists")
        return
    
    print("Running generate_list_data.py...")
    subprocess.run([sys.executable, 'src/data_generation/generate_list_data.py'], check=True)
    print("✓ List operations data generated")


def generate_grid_data():
    """Generate synthetic grid dataset"""
    print("\n" + "="*60)
    print("Generating Synthetic Grid Dataset")
    print("="*60)
    
    if Path('data/synthetic_grid_data').exists():
        print("✓ synthetic_grid_data already exists")
        return
    
    print("Running generate_grid_data.py...")
    subprocess.run([sys.executable, 'src/data_generation/generate_grid_data.py'], check=True)
    print("✓ Synthetic grid data generated")


def setup_arc_data():
    """Setup ARC dataset (create sample tasks)"""
    print("\n" + "="*60)
    print("Setting Up ARC Dataset")
    print("="*60)
    
    if Path('data/arc_data/training').exists():
        print("✓ arc_data already exists")
        return
    
    print("Running arc_data.py to create sample tasks...")
    subprocess.run([sys.executable, 'src/data_generation/arc_data.py'], check=True)
    
    print("\n" + "="*60)
    print("ARC Data Setup Instructions")
    print("="*60)
    print("Sample tasks created. For full ARC dataset:")
    print("1. Clone: git clone https://github.com/fchollet/ARC-AGI.git")
    print("2. Copy: ARC-AGI/data/training/*.json -> arc_data/training/")
    print("3. Copy: ARC-AGI/data/evaluation/*.json -> arc_data/evaluation/")
    print("="*60)


def verify_data():
    """Verify all datasets are present"""
    print("\n" + "="*60)
    print("Verifying Data")
    print("="*60)
    
    datasets = {
        'List Operations': 'data/list_ops_data',
        'Synthetic Grids': 'data/synthetic_grid_data',
        'ARC (sample)': 'data/arc_data/training'
    }
    
    all_present = True
    for name, path in datasets.items():
        if Path(path).exists():
            # Count files
            if Path(path).is_dir():
                num_files = len(list(Path(path).glob('*.json')))
                print(f"  ✓ {name}: {num_files} files")
            else:
                print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - MISSING")
            all_present = False
    
    return all_present


def main():
    print("="*60)
    print("LPN Setup - Data Generation")
    print("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Generate datasets
    generate_list_data()
    generate_grid_data()
    setup_arc_data()
    
    # Verify everything
    if verify_data():
        print("\n" + "="*60)
        print("✓ Setup Complete!")
        print("="*60)
        print("\nYou can now run:")
        print("  python run_experiments.py --experiment baseline")
        print("  python run_experiments.py --experiment poe")
        print("  python run_experiments.py --experiment spatial")
        print("  python run_experiments.py --experiment all")
    else:
        print("\n⚠ Some datasets are missing")
        print("Please check the errors above")


if __name__ == "__main__":
    main()
