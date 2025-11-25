"""
Master Run Script
Executes the full Bayesian LPN pipeline
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"Command: {cmd}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed with error code {e.returncode}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ {description}: {filepath} ({size:.2f} MB)")
        return True
    else:
        print(f"✗ {description} not found: {filepath}")
        return False


def run_poe_track():
    """Run Product of Experts track"""
    print("\n" + "=" * 80)
    print("TRACK 1: PRODUCT OF EXPERTS ON LIST OPERATIONS")
    print("=" * 80)
    
    # Check if data exists
    if not check_file_exists('./list_ops_data/train.json', 'List operations data'):
        print("\nGenerating list operations data...")
        if not run_command('python generate_list_data.py', 'Generate list operations data'):
            return False
    
    # Train PoE
    if not run_command('python train_poe.py --num_epochs 50 --batch_size 32', 
                      'Train Product of Experts LPN'):
        return False
    
    # Test PoE
    if not run_command('python test_poe.py', 
                      'Test Product of Experts LPN'):
        return False
    
    # Compare methods
    if check_file_exists('results/models/best_lpn_model.pt', 'Baseline model'):
        if not run_command('python compare_methods.py', 
                          'Compare PoE vs Baseline'):
            return False
    else:
        print("\n⚠ Skipping comparison (baseline model not found)")
        print("  Run 'python train_lpn.py' first to train baseline")
    
    print("\n" + "=" * 80)
    print("✓ TRACK 1 COMPLETE")
    print("=" * 80)
    print("\nResults saved:")
    print("  - best_poe_model.pt")
    print("  - poe_training_history.json")
    print("  - poe_test_results.json")
    if Path('results/metrics/comparison_results.json').exists():
        print("  - comparison_results.json")
    
    return True


def run_spatial_track():
    """Run Spatial LPN track"""
    print("\n" + "=" * 80)
    print("TRACK 2: SPATIAL LPN ON GRID TASKS")
    print("=" * 80)
    
    # Generate grid data
    if not check_file_exists('./synthetic_grid_data/train.json', 'Synthetic grid data'):
        print("\nGenerating synthetic grid data...")
        if not run_command('python generate_grid_data.py', 
                          'Generate synthetic grid tasks'):
            return False
    
    # Train spatial model
    if not run_command('python train_spatial.py --num_epochs 30 --batch_size 16', 
                      'Train Spatial LPN'):
        return False
    
    # Test spatial model
    if not run_command('python test_spatial.py --visualize', 
                      'Test Spatial LPN with visualization'):
        return False
    
    print("\n" + "=" * 80)
    print("✓ TRACK 2 COMPLETE")
    print("=" * 80)
    print("\nResults saved:")
    print("  - best_spatial_model.pt")
    print("  - spatial_training_history.json")
    print("  - spatial_test_results.json")
    
    return True


def quick_test():
    """Quick test that everything is set up correctly"""
    print("\n" + "=" * 80)
    print("QUICK SETUP TEST")
    print("=" * 80)
    
    print("\nChecking Python packages...")
    required = ['torch', 'numpy', 'tqdm', 'matplotlib', 'seaborn']
    all_good = True
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not installed")
            all_good = False
    
    if not all_good:
        print("\n⚠ Missing packages. Install with:")
        print("  pip install torch numpy tqdm matplotlib seaborn")
        return False
    
    print("\nChecking model files...")
    models = {
        'poe_model.py': 'Product of Experts model',
        'spatial_model.py': 'Spatial LPN model',
        'src/training/train_poe.py': 'PoE training script',
        'src/training/train_spatial.py': 'Spatial training script',
        'src/data_generation/generate_grid_data.py': 'Grid data generator'
    }
    
    for file, desc in models.items():
        check_file_exists(file, desc)
    
    print("\n✓ Setup looks good!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run Bayesian LPN Pipeline')
    parser.add_argument('--track', choices=['poe', 'spatial', 'both'], default='both',
                       help='Which track to run')
    parser.add_argument('--test-only', action='store_true',
                       help='Just test setup, don\'t train')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BAYESIAN LPN MASTER SCRIPT")
    print("=" * 80)
    print(f"Track: {args.track}")
    print(f"Test only: {args.test_only}")
    
    # Quick test
    if not quick_test():
        print("\n✗ Setup test failed")
        return 1
    
    if args.test_only:
        print("\n✓ Test-only mode - exiting")
        return 0
    
    # Run selected tracks
    success = True
    
    if args.track in ['poe', 'both']:
        if not run_poe_track():
            print("\n✗ PoE track failed")
            success = False
    
    if args.track in ['spatial', 'both']:
        if not run_spatial_track():
            print("\n✗ Spatial track failed")
            success = False
    
    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("✓ ALL TRACKS COMPLETED SUCCESSFULLY")
    else:
        print("✗ SOME TRACKS FAILED")
    print("=" * 80)
    
    print("\nNext steps:")
    print("1. Review results in JSON files")
    print("2. Run 'python analyze_poe_results.py' for visualizations")
    print("3. Try experiments from RUNNING_THE_CODE.md")
    print("4. Move to ARC dataset when ready")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
