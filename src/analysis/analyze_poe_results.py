"""
Analysis and Visualization for Product of Experts Results
Creates plots comparing PoE vs baseline
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results():
    """Load all result files"""
    results = {}
    
    files = {
        'poe_training': 'results/metrics/poe_training_history.json',
        'poe_test': 'results/metrics/poe_test_results.json',
        'comparison': 'results/metrics/comparison_results.json',
        'baseline_training': 'results/metrics/training_history.json'
    }
    
    for name, filepath in files.items():
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                results[name] = json.load(f)
            print(f"✓ Loaded {filepath}")
        else:
            print(f"⚠ {filepath} not found")
    
    return results


def plot_training_curves(results, save_dir='analysis_outputs'):
    """Plot training curves for PoE and baseline"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    ax = axes[0, 0]
    if 'poe_training' in results:
        poe = results['poe_training']
        ax.plot(poe['train_loss'], label='PoE Train Loss', linewidth=2)
    if 'baseline_training' in results:
        base = results['baseline_training']
        ax.plot(base['train_loss'], label='Baseline Train Loss', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Validation accuracy
    ax = axes[0, 1]
    if 'poe_training' in results:
        ax.plot(poe['val_accuracy'], label='PoE Val Accuracy', linewidth=2)
    if 'baseline_training' in results:
        ax.plot(base['val_accuracy'], label='Baseline Val Accuracy', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Agreement score (PoE only)
    ax = axes[1, 0]
    if 'poe_training' in results and 'train_agreement' in poe:
        ax.plot(poe['train_agreement'], label='Training Agreement', linewidth=2, color='green')
        ax.plot(poe['val_agreement'], label='Val Agreement', linewidth=2, color='orange')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Agreement Score')
        ax.set_title('PoE Agreement Score (Higher = Better Consistency)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # KL divergence
    ax = axes[1, 1]
    if 'poe_training' in results:
        ax.plot(poe['train_kl'], label='PoE KL Div', linewidth=2)
    if 'baseline_training' in results:
        ax.plot(base['train_kl'], label='Baseline KL Div', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence (Regularization)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/poe_training_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_method_comparison(results, save_dir='analysis_outputs'):
    """Plot comparison of all methods"""
    if 'comparison' not in results:
        print("⚠ No comparison results found")
        return
    
    comp = results['comparison']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax = axes[0]
    methods = ['baseline_no_search', 'baseline_with_search', 'poe_no_search', 'poe_with_search']
    labels = ['Baseline\n(no search)', 'Baseline\n(with search)', 'PoE\n(no search)', 'PoE\n(with search)']
    accuracies = [comp[m]['accuracy'] for m in methods]
    colors = ['skyblue', 'steelblue', 'lightcoral', 'darkred']
    
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Method Comparison: Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison
    ax = axes[1]
    mses = [comp[m]['mse'] for m in methods]
    bars = ax.bar(labels, mses, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('MSE (Lower = Better)', fontsize=12)
    ax.set_title('Method Comparison: MSE', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = f'{save_dir}/method_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_improvement_analysis(results, save_dir='analysis_outputs'):
    """Plot improvement from PoE and search"""
    if 'comparison' not in results:
        return
    
    comp = results['comparison']
    
    baseline_no = comp['baseline_no_search']['accuracy']
    baseline_with = comp['baseline_with_search']['accuracy']
    poe_no = comp['poe_no_search']['accuracy']
    poe_with = comp['poe_with_search']['accuracy']
    
    # Calculate improvements
    poe_improvement = (poe_no - baseline_no) / baseline_no * 100
    search_baseline = (baseline_with - baseline_no) / baseline_no * 100
    search_poe = (poe_with - poe_no) / poe_no * 100
    total_improvement = (poe_with - baseline_no) / baseline_no * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = [poe_improvement, search_baseline, search_poe, total_improvement]
    labels = ['PoE over\nBaseline', 'Search helps\nBaseline', 'Search helps\nPoE', 'Total\nImprovement']
    colors = ['coral', 'steelblue', 'steelblue', 'darkgreen']
    
    bars = ax.bar(labels, improvements, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Improvement Analysis', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    save_path = f'{save_dir}/improvement_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_per_program_comparison(results, save_dir='analysis_outputs'):
    """Plot per-program comparison"""
    if 'poe_test' not in results:
        return
    
    poe_test = results['poe_test']
    
    # Get top 15 programs by improvement
    improvements = []
    for prog in poe_test['no_search']['per_program'].keys():
        no_s = poe_test['no_search']['per_program'][prog]['accuracy']
        with_s = poe_test['with_search']['per_program'][prog]['accuracy']
        improvements.append((prog, no_s, with_s, with_s - no_s))
    
    improvements.sort(key=lambda x: x[3], reverse=True)
    top_15 = improvements[:15]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    programs = [x[0] for x in top_15]
    no_search = [x[1] for x in top_15]
    with_search = [x[2] for x in top_15]
    
    x = np.arange(len(programs))
    width = 0.35
    
    ax.barh(x - width/2, no_search, width, label='No Search', alpha=0.8, color='lightcoral')
    ax.barh(x + width/2, with_search, width, label='With Search', alpha=0.8, color='darkred')
    
    ax.set_yticks(x)
    ax.set_yticklabels(programs, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Top 15 Programs: PoE Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/per_program_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def generate_summary_report(results, save_dir='analysis_outputs'):
    """Generate text summary report"""
    Path(save_dir).mkdir(exist_ok=True)
    
    report = []
    report.append("=" * 80)
    report.append("PRODUCT OF EXPERTS - ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Training summary
    if 'poe_training' in results:
        poe = results['poe_training']
        report.append("TRAINING RESULTS:")
        report.append(f"  Final train loss: {poe['train_loss'][-1]:.4f}")
        report.append(f"  Final val accuracy: {poe['val_accuracy'][-1]:.4f}")
        if 'train_agreement' in poe:
            report.append(f"  Final agreement score: {poe['train_agreement'][-1]:.3f}")
        report.append("")
    
    # Test summary
    if 'poe_test' in results:
        poe_test = results['poe_test']
        no_s = poe_test['no_search']['overall']
        with_s = poe_test['with_search']['overall']
        
        report.append("TEST RESULTS:")
        report.append(f"  Without search:")
        report.append(f"    Accuracy: {no_s['accuracy']:.4f}")
        report.append(f"    MSE: {no_s['mse']:.4f}")
        if 'agreement' in no_s:
            report.append(f"    Agreement: {no_s['agreement']:.3f}")
        report.append(f"  With search:")
        report.append(f"    Accuracy: {with_s['accuracy']:.4f}")
        report.append(f"    MSE: {with_s['mse']:.4f}")
        
        imp = (with_s['accuracy'] - no_s['accuracy']) / no_s['accuracy'] * 100
        report.append(f"  Improvement from search: {imp:+.2f}%")
        report.append("")
    
    # Comparison summary
    if 'comparison' in results:
        comp = results['comparison']
        report.append("COMPARISON WITH BASELINE:")
        
        baseline_no = comp['baseline_no_search']['accuracy']
        poe_no = comp['poe_no_search']['accuracy']
        poe_with = comp['poe_with_search']['accuracy']
        
        poe_imp = (poe_no - baseline_no) / baseline_no * 100
        total_imp = (poe_with - baseline_no) / baseline_no * 100
        
        report.append(f"  Baseline (no search): {baseline_no:.4f}")
        report.append(f"  PoE (no search): {poe_no:.4f} ({poe_imp:+.2f}%)")
        report.append(f"  PoE (with search): {poe_with:.4f} ({total_imp:+.2f}%)")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    if 'comparison' in results:
        if poe_imp > 5:
            report.append("  ✓ PoE provides significant improvement (>5%)")
            report.append("    → Use PoE for production")
        elif poe_imp > 2:
            report.append("  ✓ PoE provides modest improvement (2-5%)")
            report.append("    → Use PoE if computational cost acceptable")
        else:
            report.append("  ⚠ PoE provides minimal improvement (<2%)")
            report.append("    → Consider tuning hyperparameters")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    save_path = f'{save_dir}/poe_summary_report.txt'
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"✓ Saved {save_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    print("=" * 80)
    print("Product of Experts - Results Analysis")
    print("=" * 80)
    print()
    
    # Load results
    print("Loading results...")
    results = load_results()
    print()
    
    if not results:
        print("⚠ No results found. Run training/testing first.")
        return
    
    # Create visualizations
    print("Generating visualizations...")
    plot_training_curves(results)
    plot_method_comparison(results)
    plot_improvement_analysis(results)
    plot_per_program_comparison(results)
    
    # Generate summary
    print("\nGenerating summary report...")
    generate_summary_report(results)
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete! Check analysis_outputs/ directory")
    print("=" * 80)


if __name__ == "__main__":
    main()
