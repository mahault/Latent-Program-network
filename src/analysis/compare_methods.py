"""
Compare Product of Experts LPN vs Baseline LPN
Side-by-side evaluation on the same test set
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
import json

from src.models.poe_model import ProductOfExpertsLPN
from src.models.lpn_model import LatentProgramNetwork, ListOpsDataset


def evaluate_model(model, dataloader, device, use_search=False, model_type='baseline'):
    """Evaluate any LPN model"""
    model.eval()
    
    total_mse = 0
    total_exact = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            train_masks = batch['train_masks'].to(device)
            test_inputs = batch['test_inputs'][:, 0, :].to(device)
            test_outputs = batch['test_outputs'][:, 0, :].to(device)
            test_masks = batch['test_masks'][:, 0, :].to(device)
            
            if use_search:
                predictions = model.predict_with_search(
                    train_inputs, train_outputs, train_masks, test_inputs,
                    num_steps=50, lr=0.1
                )
            else:
                if model_type == 'poe':
                    predictions, _, _, _ = model(train_inputs, train_outputs, test_inputs)
                else:
                    predictions, _, _ = model(train_inputs, train_outputs, test_inputs)
            
            # Metrics
            masked_pred = predictions * test_masks
            masked_target = test_outputs * test_masks
            
            mse = torch.nn.functional.mse_loss(masked_pred, masked_target, reduction='sum').item()
            exact = (torch.abs(masked_pred - masked_target) < 0.5).all(dim=1).sum().item()
            
            total_mse += mse
            total_exact += exact
            total_samples += test_inputs.shape[0]
    
    return {
        'mse': total_mse / total_samples,
        'accuracy': total_exact / total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Compare PoE vs Baseline LPN')
    parser.add_argument('--data_dir', type=str, default='./data/list_ops_data')
    parser.add_argument('--baseline_model', type=str, default='results/models/best_lpn_model.pt')
    parser.add_argument('--poe_model', type=str, default='results/models/best_poe_model.pt')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPARISON: Product of Experts vs Baseline LPN")
    print("=" * 70)
    
    # Load test data
    print("\nLoading test dataset...")
    test_dataset = ListOpsDataset(f"{args.data_dir}/test.json")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test size: {len(test_dataset)}")
    
    # Load baseline model
    print("\nLoading Baseline LPN...")
    baseline = LatentProgramNetwork(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_length=20
    ).to(args.device)
    baseline.load_state_dict(torch.load(args.baseline_model, map_location=args.device))
    print("✓ Baseline loaded")
    
    # Load PoE model
    print("\nLoading PoE-LPN...")
    poe = ProductOfExpertsLPN(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_length=20
    ).to(args.device)
    poe.load_state_dict(torch.load(args.poe_model, map_location=args.device))
    print("✓ PoE loaded")
    
    # Evaluate all combinations
    results = {}
    
    print("\n" + "=" * 70)
    print("1. Baseline WITHOUT search")
    print("=" * 70)
    results['baseline_no_search'] = evaluate_model(baseline, test_loader, args.device, 
                                                   use_search=False, model_type='baseline')
    
    print("\n" + "=" * 70)
    print("2. Baseline WITH search")
    print("=" * 70)
    results['baseline_with_search'] = evaluate_model(baseline, test_loader, args.device,
                                                     use_search=True, model_type='baseline')
    
    print("\n" + "=" * 70)
    print("3. PoE WITHOUT search")
    print("=" * 70)
    results['poe_no_search'] = evaluate_model(poe, test_loader, args.device,
                                             use_search=False, model_type='poe')
    
    print("\n" + "=" * 70)
    print("4. PoE WITH search")
    print("=" * 70)
    results['poe_with_search'] = evaluate_model(poe, test_loader, args.device,
                                               use_search=True, model_type='poe')
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Method':<30} {'Accuracy':<15} {'MSE':<15} {'vs Baseline No Search':<20}")
    print("-" * 90)
    
    baseline_no_search_acc = results['baseline_no_search']['accuracy']
    
    for name, res in results.items():
        improvement = (res['accuracy'] - baseline_no_search_acc) / baseline_no_search_acc * 100
        print(f"{name:<30} {res['accuracy']:<15.4f} {res['mse']:<15.4f} {improvement:+.2f}%")
    
    # Key insights
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    
    # PoE improvement over baseline (no search)
    poe_imp = (results['poe_no_search']['accuracy'] - results['baseline_no_search']['accuracy']) / results['baseline_no_search']['accuracy'] * 100
    print(f"1. PoE improves over baseline (no search): {poe_imp:+.2f}%")
    
    # Search improvement for baseline
    baseline_search_imp = (results['baseline_with_search']['accuracy'] - results['baseline_no_search']['accuracy']) / results['baseline_no_search']['accuracy'] * 100
    print(f"2. Search improves baseline: {baseline_search_imp:+.2f}%")
    
    # Search improvement for PoE
    poe_search_imp = (results['poe_with_search']['accuracy'] - results['poe_no_search']['accuracy']) / results['poe_no_search']['accuracy'] * 100
    print(f"3. Search improves PoE: {poe_search_imp:+.2f}%")
    
    # Best overall
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"4. Best method: {best_method[0]} ({best_method[1]['accuracy']:.4f})")
    
    # Total improvement
    total_imp = (best_method[1]['accuracy'] - baseline_no_search_acc) / baseline_no_search_acc * 100
    print(f"5. Total improvement over baseline: {total_imp:+.2f}%")
    
    # Save results
    with open('results/metrics/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved comparison results to comparison_results.json")
    
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)
    if poe_imp > 5:
        print("✓ Product of Experts provides SIGNIFICANT improvement (>5%)")
        print("  → Use PoE for production")
    elif poe_imp > 2:
        print("✓ Product of Experts provides modest improvement (2-5%)")
        print("  → Use PoE if computational cost is acceptable")
    else:
        print("⚠ Product of Experts provides minimal improvement (<2%)")
        print("  → Baseline may be sufficient, or PoE needs tuning")
    
    if poe_search_imp > baseline_search_imp:
        print("✓ PoE benefits MORE from test-time search than baseline")
        print("  → PoE latent space is more amenable to optimization")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
