"""
Testing script for Latent Program Network
Evaluates model performance with and without test-time search
"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import Dict

from lpn_model import (
    LatentProgramNetwork, 
    ListOpsDataset,
    evaluate
)


def detailed_evaluation(model: LatentProgramNetwork, dataloader: DataLoader, 
                       device: str, use_search: bool = False) -> Dict:
    """
    Detailed evaluation with per-program-type breakdown
    """
    model.eval()
    
    # Track metrics per program type
    program_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'mse': 0.0})
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing ({'with' if use_search else 'no'} search)"):
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            train_masks = batch['train_masks'].to(device)
            test_inputs = batch['test_inputs'][:, 0, :].to(device)
            test_outputs = batch['test_outputs'][:, 0, :].to(device)
            test_masks = batch['test_masks'][:, 0, :].to(device)
            program_types = batch['program_type']
            
            if use_search:
                predictions = model.predict_with_search(
                    train_inputs, train_outputs, train_masks, test_inputs,
                    num_steps=50, lr=0.1
                )
            else:
                mu, logvar = model.encoder(train_inputs, train_outputs)
                predictions = model.decoder(mu, test_inputs)
            
            # Process each sample in batch
            for i in range(predictions.shape[0]):
                pred = predictions[i] * test_masks[i]
                target = test_outputs[i] * test_masks[i]
                prog_type = program_types[i]
                
                # MSE
                mse = torch.mean((pred - target) ** 2).item()
                program_metrics[prog_type]['mse'] += mse
                
                # Exact match (with tolerance)
                exact = (torch.abs(pred - target) < 0.5).all().item()
                program_metrics[prog_type]['correct'] += int(exact)
                program_metrics[prog_type]['total'] += 1
                
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
    
    # Compute aggregate metrics
    results = {
        'per_program': {},
        'overall': {
            'accuracy': 0,
            'mse': 0
        }
    }
    
    total_correct = 0
    total_samples = 0
    total_mse = 0
    
    for prog_type, metrics in program_metrics.items():
        acc = metrics['correct'] / metrics['total']
        mse = metrics['mse'] / metrics['total']
        
        results['per_program'][prog_type] = {
            'accuracy': acc,
            'mse': mse,
            'num_samples': metrics['total']
        }
        
        total_correct += metrics['correct']
        total_samples += metrics['total']
        total_mse += metrics['mse']
    
    results['overall']['accuracy'] = total_correct / total_samples
    results['overall']['mse'] = total_mse / total_samples
    
    return results


def compare_with_without_search(model: LatentProgramNetwork, test_loader: DataLoader, 
                                device: str) -> Dict:
    """
    Compare performance with and without test-time search
    """
    print("\n" + "=" * 60)
    print("Evaluating WITHOUT test-time search...")
    print("=" * 60)
    results_no_search = detailed_evaluation(model, test_loader, device, use_search=False)
    
    print("\n" + "=" * 60)
    print("Evaluating WITH test-time search...")
    print("=" * 60)
    results_with_search = detailed_evaluation(model, test_loader, device, use_search=True)
    
    return {
        'no_search': results_no_search,
        'with_search': results_with_search
    }


def print_results_table(results: Dict):
    """Print a formatted table of results"""
    
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"{'Metric':<20} {'No Search':<20} {'With Search':<20} {'Improvement':<20}")
    print("-" * 80)
    
    no_search = results['no_search']['overall']
    with_search = results['with_search']['overall']
    
    # Accuracy
    acc_imp = (with_search['accuracy'] - no_search['accuracy']) / no_search['accuracy'] * 100
    print(f"{'Accuracy':<20} {no_search['accuracy']:<20.4f} {with_search['accuracy']:<20.4f} {acc_imp:+.2f}%")
    
    # MSE
    mse_imp = (no_search['mse'] - with_search['mse']) / no_search['mse'] * 100
    print(f"{'MSE':<20} {no_search['mse']:<20.4f} {with_search['mse']:<20.4f} {mse_imp:+.2f}%")
    
    print("\n" + "=" * 80)
    print("PER-PROGRAM-TYPE RESULTS (Top 15 by improvement)")
    print("=" * 80)
    print(f"{'Program Type':<25} {'No Search':<15} {'With Search':<15} {'Improvement':<15}")
    print("-" * 80)
    
    # Calculate improvements
    improvements = []
    for prog_type in results['no_search']['per_program'].keys():
        no_s = results['no_search']['per_program'][prog_type]['accuracy']
        with_s = results['with_search']['per_program'][prog_type]['accuracy']
        imp = with_s - no_s
        improvements.append((prog_type, no_s, with_s, imp))
    
    # Sort by improvement
    improvements.sort(key=lambda x: x[3], reverse=True)
    
    # Print top 15
    for prog_type, no_s, with_s, imp in improvements[:15]:
        print(f"{prog_type:<25} {no_s:<15.4f} {with_s:<15.4f} {imp:+.4f}")
    
    # Print bottom 5 (worst performing)
    if len(improvements) > 15:
        print("\n" + "-" * 80)
        print("Bottom 5 (least improvement):")
        print("-" * 80)
        for prog_type, no_s, with_s, imp in improvements[-5:]:
            print(f"{prog_type:<25} {no_s:<15.4f} {with_s:<15.4f} {imp:+.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test Latent Program Network')
    parser.add_argument('--data_dir', type=str, default='./list_ops_data',
                      help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='best_lpn_model.pt',
                      help='Path to trained model')
    parser.add_argument('--latent_dim', type=int, default=64,
                      help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension for networks')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to test on')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Latent Program Network")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_path}")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ListOpsDataset(f"{args.data_dir}/test.json")
    print(f"Test size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print("\nLoading model...")
    model = LatentProgramNetwork(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_length=20
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("✓ Model loaded successfully")
    
    # Run comparison
    results = compare_with_without_search(model, test_loader, args.device)
    
    # Print results
    print_results_table(results)
    
    # Save results
    results_path = Path('./test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed results to {results_path}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    print("Next step: Run 'python analyze_results.py' to visualize results")


if __name__ == "__main__":
    main()