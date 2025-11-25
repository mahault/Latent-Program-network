"""
Testing script for Spatial LPN
Evaluates on synthetic grid tasks with and without test-time search
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from spatial_model import SpatialLPN
from generate_grid_data import SyntheticGridDataset


def detailed_evaluation_spatial(model: SpatialLPN, dataloader: DataLoader,
                                device: str, use_search: bool = False) -> dict:
    """Detailed evaluation with per-transformation breakdown"""
    model.eval()
    
    transform_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'pixel_acc': []})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing ({'with' if use_search else 'no'} search)"):
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            test_input = batch['test_input'].to(device)
            test_output = batch['test_output'].to(device)
            transformations = batch['transformation']
            
            if use_search:
                output_logits = model.predict_with_search(
                    train_inputs, train_outputs, test_input,
                    num_steps=30, lr=0.05
                )
            else:
                output_logits, _, _ = model(train_inputs, train_outputs, test_input)
            
            # Predictions
            predictions = output_logits.argmax(dim=1)
            
            # Process each sample
            for i in range(predictions.shape[0]):
                pred = predictions[i]
                target = test_output[i]
                transform = transformations[i]
                
                # Exact match
                exact = (pred == target).all().item()
                transform_metrics[transform]['correct'] += int(exact)
                transform_metrics[transform]['total'] += 1
                
                # Pixel accuracy
                pixel_acc = (pred == target).float().mean().item()
                transform_metrics[transform]['pixel_acc'].append(pixel_acc)
    
    # Aggregate results
    results = {
        'per_transformation': {},
        'overall': {
            'accuracy': 0,
            'pixel_accuracy': 0
        }
    }
    
    total_correct = 0
    total_samples = 0
    total_pixel_acc = 0
    
    for transform, metrics in transform_metrics.items():
        acc = metrics['correct'] / metrics['total']
        pixel_acc = np.mean(metrics['pixel_acc'])
        
        results['per_transformation'][transform] = {
            'accuracy': acc,
            'pixel_accuracy': pixel_acc,
            'num_samples': metrics['total']
        }
        
        total_correct += metrics['correct']
        total_samples += metrics['total']
        total_pixel_acc += sum(metrics['pixel_acc'])
    
    results['overall']['accuracy'] = total_correct / total_samples
    results['overall']['pixel_accuracy'] = total_pixel_acc / total_samples
    
    return results


def visualize_predictions(model: SpatialLPN, dataset: SyntheticGridDataset,
                         device: str, num_samples: int = 5, use_search: bool = False):
    """Visualize some predictions"""
    model.eval()
    
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            
            train_inputs = sample['train_inputs'].unsqueeze(0).to(device)
            train_outputs = sample['train_outputs'].unsqueeze(0).to(device)
            test_input = sample['test_input'].unsqueeze(0).to(device)
            test_output = sample['test_output']
            
            if use_search:
                output_logits = model.predict_with_search(
                    train_inputs, train_outputs, test_input,
                    num_steps=30, lr=0.05
                )
            else:
                output_logits, _, _ = model(train_inputs, train_outputs, test_input)
            
            prediction = output_logits.argmax(dim=1)[0].cpu()
            
            # Calculate accuracy
            correct = (prediction == test_output).all().item()
            pixel_acc = (prediction == test_output).float().mean().item()
            
            print(f"\nTask: {sample['task_id']} ({sample['transformation']})")
            print(f"Correct: {'✓' if correct else '✗'} | Pixel Acc: {pixel_acc:.3f}")
            
            # Show 8x8 corner
            print("Input (8x8 corner):")
            print(test_input[0, :8, :8].cpu().numpy())
            print("Target (8x8 corner):")
            print(test_output[:8, :8].numpy())
            print("Prediction (8x8 corner):")
            print(prediction[:8, :8].numpy())
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test Spatial LPN')
    parser.add_argument('--data_dir', type=str, default='./synthetic_grid_data')
    parser.add_argument('--model_path', type=str, default='best_spatial_model.pt')
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--visualize', action='store_true', help='Show sample predictions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Spatial LPN on Grid Tasks")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_path}")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = SyntheticGridDataset(f"{args.data_dir}/test.json")
    print(f"Test size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    print("\nLoading Spatial LPN...")
    model = SpatialLPN(
        num_colors=args.num_colors,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("✓ Model loaded successfully")
    
    # Evaluate without search
    print("\n" + "=" * 60)
    print("Evaluating WITHOUT test-time search...")
    print("=" * 60)
    results_no_search = detailed_evaluation_spatial(model, test_loader, args.device, use_search=False)
    
    # Evaluate with search
    print("\n" + "=" * 60)
    print("Evaluating WITH test-time search...")
    print("=" * 60)
    results_with_search = detailed_evaluation_spatial(model, test_loader, args.device, use_search=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    no_search = results_no_search['overall']
    with_search = results_with_search['overall']
    
    print(f"\n{'Metric':<25} {'No Search':<20} {'With Search':<20} {'Improvement':<15}")
    print("-" * 80)
    
    # Grid accuracy (exact match)
    acc_imp = (with_search['accuracy'] - no_search['accuracy']) / (no_search['accuracy'] + 1e-8) * 100
    print(f"{'Grid Accuracy':<25} {no_search['accuracy']:<20.4f} {with_search['accuracy']:<20.4f} {acc_imp:+.2f}%")
    
    # Pixel accuracy
    pix_imp = (with_search['pixel_accuracy'] - no_search['pixel_accuracy']) / (no_search['pixel_accuracy'] + 1e-8) * 100
    print(f"{'Pixel Accuracy':<25} {no_search['pixel_accuracy']:<20.4f} {with_search['pixel_accuracy']:<20.4f} {pix_imp:+.2f}%")
    
    # Save results
    results = {
        'no_search': results_no_search,
        'with_search': results_with_search
    }
    
    with open('spatial_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to spatial_test_results.json")
    
    # Show per-transformation breakdown
    print("\n" + "=" * 80)
    print("PER-TRANSFORMATION RESULTS (WITHOUT SEARCH)")
    print("=" * 80)
    
    transformations = sorted(results_no_search['per_transformation'].items(),
                            key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Transformation':<20} {'Accuracy':<15} {'Pixel Acc':<15} {'Samples':<10}")
    print("-" * 80)
    for transform, metrics in transformations:
        print(f"{transform:<20} {metrics['accuracy']:<15.4f} {metrics['pixel_accuracy']:<15.4f} {metrics['num_samples']:<10}")
    
    # Show improvements
    print("\n" + "=" * 80)
    print("TOP 5 MOST IMPROVED TRANSFORMATIONS (with search)")
    print("=" * 80)
    
    improvements = []
    for transform in results_no_search['per_transformation'].keys():
        no_s = results_no_search['per_transformation'][transform]['accuracy']
        with_s = results_with_search['per_transformation'][transform]['accuracy']
        improvements.append((transform, no_s, with_s, with_s - no_s))
    
    improvements.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'Transformation':<20} {'No Search':<15} {'With Search':<15} {'Δ':<10}")
    print("-" * 80)
    for transform, no_s, with_s, imp in improvements[:5]:
        print(f"{transform:<20} {no_s:<15.4f} {with_s:<15.4f} {imp:+.4f}")
    
    # Visualize if requested
    if args.visualize:
        visualize_predictions(model, test_dataset, args.device, num_samples=5, use_search=True)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if no_search['accuracy'] > 0.8:
        print("✓ Model performs well on synthetic tasks (>80%)")
        print("  → Ready to try on ARC dataset")
    elif no_search['accuracy'] > 0.5:
        print("⚠ Model has moderate performance (50-80%)")
        print("  → Consider training longer or tuning hyperparameters")
    else:
        print("✗ Model struggles on synthetic tasks (<50%)")
        print("  → Check model architecture or training setup")
    
    if acc_imp > 10:
        print("✓ Test-time search provides significant improvement (>10%)")
    elif acc_imp > 0:
        print("⚠ Test-time search provides modest improvement")
    else:
        print("⚠ Test-time search does not help")
        print("  → Latent space may not be smooth enough")


if __name__ == "__main__":
    main()
