"""
Master Experiment Runner
Orchestrates all LPN experiments with a single command
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and report status"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def experiment_baseline(args):
    """Run baseline LPN experiment"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline LPN (List Operations)")
    print("="*70)
    
    # Train
    if not run_command(
        [sys.executable, 'train_lpn.py', 
         '--num_epochs', str(args.epochs),
         '--batch_size', str(args.batch_size)],
        "Training Baseline LPN"
    ):
        return False
    
    # Test
    if not run_command(
        [sys.executable, 'test_lpn.py'],
        "Testing Baseline LPN"
    ):
        return False
    
    # Analyze
    if not run_command(
        [sys.executable, 'analyze_results.py'],
        "Analyzing Baseline Results"
    ):
        return False
    
    return True


def experiment_poe(args):
    """Run Product of Experts LPN experiment"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Product of Experts LPN")
    print("="*70)
    
    # Train
    if not run_command(
        [sys.executable, 'train_poe.py',
         '--num_epochs', str(args.epochs),
         '--batch_size', str(args.batch_size)],
        "Training PoE-LPN"
    ):
        return False
    
    # Test
    if not run_command(
        [sys.executable, 'test_poe.py'],
        "Testing PoE-LPN"
    ):
        return False
    
    return True


def experiment_compare(args):
    """Compare baseline vs PoE"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Comparison (Baseline vs PoE)")
    print("="*70)
    
    # Check models exist
    if not Path('best_lpn_model.pt').exists():
        print("✗ Baseline model not found. Run experiment 1 first.")
        return False
    
    if not Path('best_poe_model.pt').exists():
        print("✗ PoE model not found. Run experiment 2 first.")
        return False
    
    # Compare
    if not run_command(
        [sys.executable, 'compare_methods.py'],
        "Comparing Methods"
    ):
        return False
    
    return True


def experiment_spatial(args):
    """Run spatial LPN experiment"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Spatial LPN (Grid Tasks)")
    print("="*70)
    
    # Train
    if not run_command(
        [sys.executable, 'train_spatial.py',
         '--num_epochs', str(args.epochs),
         '--batch_size', str(max(args.batch_size // 2, 8))],  # Smaller batch for grids
        "Training Spatial LPN"
    ):
        return False
    
    print("\n✓ Spatial LPN training complete")
    print("Note: Testing spatial LPN requires manual evaluation")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run LPN Experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['baseline', 'poe', 'compare', 'spatial', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with fewer epochs (10)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.epochs = 10
        print("Quick mode: Using 10 epochs")
    
    print("="*70)
    print("LPN Experiment Runner")
    print("="*70)
    print(f"Experiment: {args.experiment}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    # Check data exists
    if not Path('list_ops_data').exists():
        print("\n✗ Data not found! Please run: python setup.py")
        return
    
    # Run experiments
    start_time = time.time()
    success = False
    
    if args.experiment == 'baseline':
        success = experiment_baseline(args)
    
    elif args.experiment == 'poe':
        success = experiment_poe(args)
    
    elif args.experiment == 'compare':
        success = experiment_compare(args)
    
    elif args.experiment == 'spatial':
        success = experiment_spatial(args)
    
    elif args.experiment == 'all':
        print("\nRunning all experiments...")
        
        # 1. Baseline
        if not experiment_baseline(args):
            print("\n✗ Baseline experiment failed")
            return
        
        # 2. PoE
        if not experiment_poe(args):
            print("\n✗ PoE experiment failed")
            return
        
        # 3. Compare
        if not experiment_compare(args):
            print("\n✗ Comparison failed")
            return
        
        # 4. Spatial
        if not experiment_spatial(args):
            print("\n✗ Spatial experiment failed")
            return
        
        success = True
    
    # Final report
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    if success:
        print(f"✓ All experiments completed in {elapsed/60:.1f} minutes")
        print("="*70)
        
        print("\nResults:")
        if Path('analysis_outputs').exists():
            print("  📊 Visualizations: analysis_outputs/")
        if Path('test_results.json').exists():
            print("  📝 Baseline results: test_results.json")
        if Path('poe_test_results.json').exists():
            print("  📝 PoE results: poe_test_results.json")
        if Path('comparison_results.json').exists():
            print("  📝 Comparison: comparison_results.json")
        
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("1. Check analysis_outputs/ for visualizations")
        print("2. Review test_results.json for detailed metrics")
        print("3. See BAYESIAN_APPROACH.md for theory")
        print("4. See ROADMAP.md for next phases")
    else:
        print(f"✗ Experiments failed after {elapsed/60:.1f} minutes")
        print("="*70)


if __name__ == "__main__":
    main()
