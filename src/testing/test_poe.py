"""
Testing script for Product of Experts LPN
Tests both with and without test-time search
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
from collections import defaultdict
from tqdm import tqdm
from typing import Dict

from src.models.poe_model import ProductOfExpertsLPN
from src.models.lpn_model import ListOpsDataset


def detailed_evaluation_poe(model: ProductOfExpertsLPN, dataloader: DataLoader,
                            device: str, use_search: bool = False) -> Dict:
    """Detailed evaluation with per-program breakdown"""
    model.eval()
    
    program_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'mse': 0.0, 'agreement': []})
    
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
                # Compute agreement for monitoring
                _, _, agreement = model.encode_examples(train_inputs, train_outputs)
            else:
                predictions, _, _, agreement = model(train_inputs, train_outputs, test_inputs)
            
            # Process each sample
            for i in range(predictions.shape[0]):
                pred = predictions[i] * test_masks[i]
                target = test_outputs[i] * test_masks[i]
                prog_type = program_types[i]
                
                # MSE
                mse = torch.mean((pred - target) ** 2).item()
                program_metrics[prog_type]['mse'] += mse
                
                # Exact match
                exact = (torch.abs(pred - target) < 0.5).all().item()
                program_metrics[prog_type]['correct'] += int(exact)
                program_metrics[prog_type]['total'] += 1
                
                # Agreement
                program_metrics[prog_type]['agreement'].append(agreement[i].item())
    
    # Aggregate results
    results = {
        'per_program': {},
        'overall': {
            'accuracy': 0,
            'mse': 0,
            'agreement': 0
        }
    }
    
    total_correct = 0
    total_samples = 0
    total_mse = 0
    total_agreement = 0
    
    for prog_type, metrics in program_metrics.items():
        acc = metrics['correct'] / metrics['total']
        mse = metrics['mse'] / metrics['total']
        avg_agreement = sum(metrics['agreement']) / len(metrics['agreement'])
        
        results['per_program'][prog_type] = {
            'accuracy': acc,
            'mse': mse,
            'agreement': avg_agreement,
            'num_samples': metrics['total']
        }
        
        total_correct += metrics['correct']
        total_samples += metrics['total']
        total_mse += metrics['mse']
        total_agreement += sum(metrics['agreement'])
    
    results['overall']['accuracy'] = total_correct / total_samples
    results['overall']['mse'] = total_mse / total_samples
    results['overall']['agreement'] = total_agreement / total_samples
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test Product of Experts LPN')
    parser.add_argument('--data_dir', type=str, default='./data/list_ops_data')
    parser.add_argument('--model_path', type=str, default='results/models/best_poe_model.pt')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Product of Experts LPN")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_path}")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ListOpsDataset(f"{args.data_dir}/test.json")
    print(f"Test size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    print("\nLoading PoE-LPN...")
    model = ProductOfExpertsLPN(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_length=20
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("✓ Model loaded successfully")
    
    # Evaluate without search
    print("\n" + "=" * 60)
    print("Evaluating WITHOUT test-time search...")
    print("=" * 60)
    results_no_search = detailed_evaluation_poe(model, test_loader, args.device, use_search=False)
    
    # Evaluate with search
    print("\n" + "=" * 60)
    print("Evaluating WITH test-time search...")
    print("=" * 60)
    results_with_search = detailed_evaluation_poe(model, test_loader, args.device, use_search=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    no_search = results_no_search['overall']
    with_search = results_with_search['overall']
    
    print(f"\n{'Metric':<20} {'No Search':<20} {'With Search':<20} {'Improvement':<15}")
    print("-" * 80)
    
    # Accuracy
    acc_imp = (with_search['accuracy'] - no_search['accuracy']) / no_search['accuracy'] * 100
    print(f"{'Accuracy':<20} {no_search['accuracy']:<20.4f} {with_search['accuracy']:<20.4f} {acc_imp:+.2f}%")
    
    # MSE
    mse_imp = (no_search['mse'] - with_search['mse']) / no_search['mse'] * 100
    print(f"{'MSE':<20} {no_search['mse']:<20.4f} {with_search['mse']:<20.4f} {mse_imp:+.2f}%")
    
    # Agreement
    print(f"{'Agreement':<20} {no_search['agreement']:<20.4f} {with_search['agreement']:<20.4f}")
    
    # Save results
    results = {
        'no_search': results_no_search,
        'with_search': results_with_search
    }
    
    with open('results/metrics/poe_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to poe_test_results.json")
    
    # Show top improved programs
    print("\n" + "=" * 80)
    print("TOP 10 MOST IMPROVED PROGRAMS (with test-time search)")
    print("=" * 80)
    
    improvements = []
    for prog_type in results_no_search['per_program'].keys():
        no_s = results_no_search['per_program'][prog_type]['accuracy']
        with_s = results_with_search['per_program'][prog_type]['accuracy']
        agr = results_no_search['per_program'][prog_type]['agreement']
        improvements.append((prog_type, no_s, with_s, with_s - no_s, agr))
    
    improvements.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'Program':<25} {'No Search':<12} {'With Search':<12} {'Δ':<10} {'Agreement':<10}")
    print("-" * 80)
    for prog, no_s, with_s, imp, agr in improvements[:10]:
        print(f"{prog:<25} {no_s:<12.4f} {with_s:<12.4f} {imp:+.4f}    {agr:.3f}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
