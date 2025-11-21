"""
Analysis and Visualization of LPN Results
Creates plots and detailed analysis of model performance
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")


def load_results():
    """Load training history and test results"""
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    with open('test_results.json', 'r') as f:
        test_results = json.load(f)
    
    return history, test_results


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, history['train_recon'], label='Reconstruction Loss', 
                   color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[1, 0].plot(epochs, history['train_kl'], label='KL Divergence', 
                   color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1, 1].plot(epochs, history['val_accuracy'], label='Val Accuracy', 
                   color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


def plot_test_comparison(test_results, save_path='test_comparison.png'):
    """Plot comparison of with/without search"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall metrics
    metrics = ['Accuracy', 'MSE']
    no_search_vals = [
        test_results['no_search']['overall']['accuracy'],
        test_results['no_search']['overall']['mse']
    ]
    with_search_vals = [
        test_results['with_search']['overall']['accuracy'],
        test_results['with_search']['overall']['mse']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, no_search_vals, width, label='No Search', alpha=0.8)
    axes[0].bar(x + width/2, with_search_vals, width, label='With Search', alpha=0.8)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Overall Performance: No Search vs With Search')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Per-program improvement
    improvements = []
    program_types = []
    
    for prog_type in test_results['no_search']['per_program'].keys():
        no_s = test_results['no_search']['per_program'][prog_type]['accuracy']
        with_s = test_results['with_search']['per_program'][prog_type]['accuracy']
        improvements.append(with_s - no_s)
        program_types.append(prog_type)
    
    # Sort by improvement
    sorted_indices = np.argsort(improvements)[::-1]
    top_15_indices = sorted_indices[:15]
    
    top_improvements = [improvements[i] for i in top_15_indices]
    top_programs = [program_types[i] for i in top_15_indices]
    
    colors = ['green' if x > 0 else 'red' for x in top_improvements]
    
    axes[1].barh(range(len(top_programs)), top_improvements, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(top_programs)))
    axes[1].set_yticklabels(top_programs, fontsize=8)
    axes[1].set_xlabel('Accuracy Improvement')
    axes[1].set_title('Top 15 Programs: Accuracy Improvement with Search')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved test comparison to {save_path}")
    plt.close()


def plot_program_categories(test_results, save_path='program_categories.png'):
    """Plot performance by program category"""
    
    # Categorize programs
    categories = {
        'Mapping': ['square', 'negate', 'abs', 'add_3', 'add_5', 'multiply_2', 
                   'multiply_3', 'subtract_1', 'increment', 'decrement'],
        'Filtering': ['filter_positive', 'filter_negative', 'filter_even', 
                     'filter_odd', 'filter_greater_than_5'],
        'Structural': ['reverse', 'sort_ascending', 'sort_descending', 
                      'take_first_3', 'take_last_3', 'duplicate_each', 'remove_duplicates'],
        'Reduction': ['sum', 'max', 'min', 'count', 'mean'],
        'Combination': ['cumsum', 'differences', 'alternating_sign']
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average accuracy per category
    category_accuracies_no_search = []
    category_accuracies_with_search = []
    category_names = []
    
    for cat_name, programs in categories.items():
        no_search_accs = []
        with_search_accs = []
        
        for prog in programs:
            if prog in test_results['no_search']['per_program']:
                no_search_accs.append(
                    test_results['no_search']['per_program'][prog]['accuracy']
                )
                with_search_accs.append(
                    test_results['with_search']['per_program'][prog]['accuracy']
                )
        
        if no_search_accs:
            category_accuracies_no_search.append(np.mean(no_search_accs))
            category_accuracies_with_search.append(np.mean(with_search_accs))
            category_names.append(cat_name)
    
    x = np.arange(len(category_names))
    width = 0.35
    
    axes[0].bar(x - width/2, category_accuracies_no_search, width, 
               label='No Search', alpha=0.8, color='skyblue')
    axes[0].bar(x + width/2, category_accuracies_with_search, width, 
               label='With Search', alpha=0.8, color='orange')
    axes[0].set_ylabel('Average Accuracy')
    axes[0].set_title('Performance by Program Category')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(category_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])
    
    # Improvement per category
    improvements = [with_s - no_s for no_s, with_s in 
                   zip(category_accuracies_no_search, category_accuracies_with_search)]
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    axes[1].bar(x, improvements, alpha=0.7, color=colors)
    axes[1].set_ylabel('Accuracy Improvement')
    axes[1].set_title('Improvement with Search by Category')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(category_names, rotation=45, ha='right')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved program categories plot to {save_path}")
    plt.close()


def plot_heatmap(test_results, save_path='accuracy_heatmap.png'):
    """Create heatmap of accuracies"""
    
    # Get all program types
    programs = sorted(test_results['no_search']['per_program'].keys())
    
    # Create matrix: [program, metric] where metrics are [no_search, with_search]
    data = []
    for prog in programs:
        no_s = test_results['no_search']['per_program'][prog]['accuracy']
        with_s = test_results['with_search']['per_program'][prog]['accuracy']
        data.append([no_s, with_s])
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(8, 16))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Search', 'With Search'])
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(programs, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(programs)):
        for j in range(2):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('Accuracy Heatmap: All Programs')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy heatmap to {save_path}")
    plt.close()


def generate_summary_report(history, test_results, save_path='summary_report.txt'):
    """Generate text summary report"""
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LATENT PROGRAM NETWORK - EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Training summary
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total epochs: {len(history['train_loss'])}\n")
        f.write(f"Final training loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Best validation accuracy: {max(history['val_accuracy']):.4f}\n")
        f.write("\n")
        
        # Test results
        f.write("TEST RESULTS\n")
        f.write("-" * 80 + "\n")
        
        no_search = test_results['no_search']['overall']
        with_search = test_results['with_search']['overall']
        
        f.write(f"WITHOUT TEST-TIME SEARCH:\n")
        f.write(f"  Accuracy: {no_search['accuracy']:.4f}\n")
        f.write(f"  MSE: {no_search['mse']:.4f}\n\n")
        
        f.write(f"WITH TEST-TIME SEARCH:\n")
        f.write(f"  Accuracy: {with_search['accuracy']:.4f}\n")
        f.write(f"  MSE: {with_search['mse']:.4f}\n\n")
        
        # Improvement
        acc_imp = (with_search['accuracy'] - no_search['accuracy']) / no_search['accuracy'] * 100
        mse_imp = (no_search['mse'] - with_search['mse']) / no_search['mse'] * 100
        
        f.write(f"IMPROVEMENT WITH TEST-TIME SEARCH:\n")
        f.write(f"  Accuracy: {acc_imp:+.2f}%\n")
        f.write(f"  MSE reduction: {mse_imp:+.2f}%\n\n")
        
        # Top and bottom programs
        improvements = []
        for prog_type in test_results['no_search']['per_program'].keys():
            no_s = test_results['no_search']['per_program'][prog_type]['accuracy']
            with_s = test_results['with_search']['per_program'][prog_type]['accuracy']
            improvements.append((prog_type, no_s, with_s, with_s - no_s))
        
        improvements.sort(key=lambda x: x[3], reverse=True)
        
        f.write("TOP 10 MOST IMPROVED PROGRAMS:\n")
        f.write("-" * 80 + "\n")
        for i, (prog, no_s, with_s, imp) in enumerate(improvements[:10], 1):
            f.write(f"{i:2}. {prog:<25} {no_s:.4f} → {with_s:.4f} ({imp:+.4f})\n")
        f.write("\n")
        
        f.write("TOP 10 LEAST IMPROVED PROGRAMS:\n")
        f.write("-" * 80 + "\n")
        for i, (prog, no_s, with_s, imp) in enumerate(improvements[-10:], 1):
            f.write(f"{i:2}. {prog:<25} {no_s:.4f} → {with_s:.4f} ({imp:+.4f})\n")
        f.write("\n")
        
        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n")
        if acc_imp > 5:
            f.write("✓ Test-time search provides significant improvement (>5%)\n")
        elif acc_imp > 0:
            f.write("✓ Test-time search provides modest improvement\n")
        else:
            f.write("✗ Test-time search did not improve performance\n")
        
        if max(history['val_accuracy']) > 0.7:
            f.write("✓ Model learned meaningful program representations (>70% val acc)\n")
        else:
            f.write("⚠ Model may need more training or larger capacity\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Saved summary report to {save_path}")


def main():
    print("=" * 60)
    print("Analyzing LPN Results")
    print("=" * 60)
    
    # Load data
    print("\nLoading results...")
    history, test_results = load_results()
    
    # Create output directory
    output_dir = Path('./analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training_curves(history, output_dir / 'training_curves.png')
    plot_test_comparison(test_results, output_dir / 'test_comparison.png')
    plot_program_categories(test_results, output_dir / 'program_categories.png')
    plot_heatmap(test_results, output_dir / 'accuracy_heatmap.png')
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(history, test_results, output_dir / 'summary_report.txt')
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - test_comparison.png")
    print("  - program_categories.png")
    print("  - accuracy_heatmap.png")
    print("  - summary_report.txt")


if __name__ == "__main__":
    main()
