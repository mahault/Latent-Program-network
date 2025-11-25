"""
Synthetic Grid Task Generator
Creates simple grid transformation tasks for testing spatial LPN
Similar to "unit tests" mentioned in the lab meeting
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import List, Callable, Tuple


class GridTaskGenerator:
    """
    Generates synthetic grid transformation tasks
    These are simple enough that any reasonable model should learn them
    """
    
    def __init__(self, grid_size: int = 16, num_colors: int = 10):
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        # Define transformations
        self.transformations = {
            'identity': self._identity,
            'transpose': self._transpose,
            'rotate_90': self._rotate_90,
            'rotate_180': self._rotate_180,
            'rotate_270': self._rotate_270,
            'mirror_h': self._mirror_horizontal,
            'mirror_v': self._mirror_vertical,
            'invert_colors': self._invert_colors,
            'shift_right': self._shift_right,
            'shift_down': self._shift_down,
            'add_border': self._add_border,
            'remove_background': self._remove_background,
        }
    
    # ============= Basic Transformations =============
    
    def _identity(self, grid: np.ndarray) -> np.ndarray:
        """Copy input to output"""
        return grid.copy()
    
    def _transpose(self, grid: np.ndarray) -> np.ndarray:
        """Transpose grid"""
        return grid.T
    
    def _rotate_90(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 90 degrees clockwise"""
        return np.rot90(grid, k=-1)
    
    def _rotate_180(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 180 degrees"""
        return np.rot90(grid, k=2)
    
    def _rotate_270(self, grid: np.ndarray) -> np.ndarray:
        """Rotate 270 degrees clockwise (90 counter-clockwise)"""
        return np.rot90(grid, k=1)
    
    def _mirror_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Mirror horizontally"""
        return np.fliplr(grid)
    
    def _mirror_vertical(self, grid: np.ndarray) -> np.ndarray:
        """Mirror vertically"""
        return np.flipud(grid)
    
    def _invert_colors(self, grid: np.ndarray) -> np.ndarray:
        """Invert colors: 0->9, 1->8, etc."""
        return (self.num_colors - 1) - grid
    
    def _shift_right(self, grid: np.ndarray) -> np.ndarray:
        """Shift right by 2 pixels, wrap around"""
        return np.roll(grid, shift=2, axis=1)
    
    def _shift_down(self, grid: np.ndarray) -> np.ndarray:
        """Shift down by 2 pixels, wrap around"""
        return np.roll(grid, shift=2, axis=0)
    
    def _add_border(self, grid: np.ndarray) -> np.ndarray:
        """Add colored border"""
        result = grid.copy()
        result[0, :] = 1  # Top
        result[-1, :] = 1  # Bottom
        result[:, 0] = 1  # Left
        result[:, -1] = 1  # Right
        return result
    
    def _remove_background(self, grid: np.ndarray) -> np.ndarray:
        """Set all 0s (background) to a different color"""
        result = grid.copy()
        result[grid == 0] = 5
        return result
    
    # ============= Grid Generation =============
    
    def generate_random_grid(self) -> np.ndarray:
        """Generate a random grid with simple patterns"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Add some random colored regions
        num_regions = np.random.randint(1, 5)
        
        for _ in range(num_regions):
            # Random position and size
            x = np.random.randint(0, self.grid_size - 4)
            y = np.random.randint(0, self.grid_size - 4)
            w = np.random.randint(2, 6)
            h = np.random.randint(2, 6)
            color = np.random.randint(1, self.num_colors)
            
            # Draw rectangle
            grid[y:min(y+h, self.grid_size), x:min(x+w, self.grid_size)] = color
        
        return grid
    
    def generate_task(self, transformation_name: str, num_examples: int = 3) -> dict:
        """
        Generate one task for a given transformation
        
        Returns:
            task: Dictionary with train and test examples
        """
        if transformation_name not in self.transformations:
            raise ValueError(f"Unknown transformation: {transformation_name}")
        
        transform_fn = self.transformations[transformation_name]
        
        # Generate training examples
        train_examples = []
        for _ in range(num_examples):
            input_grid = self.generate_random_grid()
            output_grid = transform_fn(input_grid)
            train_examples.append({
                'input': input_grid.tolist(),
                'output': output_grid.tolist()
            })
        
        # Generate test examples
        test_examples = []
        for _ in range(2):  # 2 test examples
            input_grid = self.generate_random_grid()
            output_grid = transform_fn(input_grid)
            test_examples.append({
                'input': input_grid.tolist(),
                'output': output_grid.tolist()
            })
        
        return {
            'transformation': transformation_name,
            'train': train_examples,
            'test': test_examples
        }
    
    def generate_dataset(self, tasks_per_transform: int = 50) -> List[dict]:
        """Generate full dataset with multiple tasks per transformation"""
        dataset = []
        
        for transform_name in self.transformations.keys():
            print(f"Generating {tasks_per_transform} tasks for '{transform_name}'...")
            
            for task_idx in range(tasks_per_transform):
                task = self.generate_task(transform_name, num_examples=3)
                task['task_id'] = f"{transform_name}_{task_idx:03d}"
                dataset.append(task)
        
        return dataset


class SyntheticGridDataset(Dataset):
    """PyTorch Dataset for synthetic grid tasks"""
    
    def __init__(self, data_path: str, max_grid_size: int = 30):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_grid_size = max_grid_size
    
    def pad_grid(self, grid: List) -> np.ndarray:
        """Pad grid to max_grid_size"""
        grid = np.array(grid, dtype=np.int32)
        h, w = grid.shape
        
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)
        padded[:h, :w] = grid
        
        return padded
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task = self.data[idx]
        
        # Training examples
        train_inputs = []
        train_outputs = []
        
        for example in task['train']:
            train_inputs.append(self.pad_grid(example['input']))
            train_outputs.append(self.pad_grid(example['output']))
        
        # Test example (first one)
        test_example = task['test'][0]
        test_input = self.pad_grid(test_example['input'])
        test_output = self.pad_grid(test_example['output'])
        
        return {
            'task_id': task['task_id'],
            'transformation': task['transformation'],
            'train_inputs': torch.tensor(np.stack(train_inputs), dtype=torch.long),
            'train_outputs': torch.tensor(np.stack(train_outputs), dtype=torch.long),
            'test_input': torch.tensor(test_input, dtype=torch.long),
            'test_output': torch.tensor(test_output, dtype=torch.long)
        }


def generate_and_save_dataset(output_dir: str = './data/synthetic_grid_data', 
                              tasks_per_transform: int = 50,
                              grid_size: int = 16):
    """Generate and save synthetic grid dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = GridTaskGenerator(grid_size=grid_size, num_colors=10)
    
    print(f"Generating synthetic grid dataset...")
    print(f"Transformations: {len(generator.transformations)}")
    print(f"Tasks per transformation: {tasks_per_transform}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print()
    
    # Generate full dataset
    all_tasks = generator.generate_dataset(tasks_per_transform)
    
    # Split into train/val/test (70/15/15)
    np.random.shuffle(all_tasks)
    n = len(all_tasks)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_tasks = all_tasks[:n_train]
    val_tasks = all_tasks[n_train:n_train + n_val]
    test_tasks = all_tasks[n_train + n_val:]
    
    # Save
    for split, tasks in [('train', train_tasks), ('val', val_tasks), ('test', test_tasks)]:
        path = output_dir / f'{split}.json'
        with open(path, 'w') as f:
            json.dump(tasks, f)
        print(f"✓ Saved {len(tasks)} tasks to {path}")
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Train: {len(train_tasks)}, Val: {len(val_tasks)}, Test: {len(test_tasks)}")
    
    return output_dir


if __name__ == "__main__":
    print("Synthetic Grid Task Generator")
    print("=" * 60)
    
    # Generate dataset
    output_dir = generate_and_save_dataset(
        output_dir='./data/synthetic_grid_data',
        tasks_per_transform=50,
        grid_size=16
    )
    
    # Test loading
    print("\nTesting dataset loading...")
    dataset = SyntheticGridDataset(f'{output_dir}/train.json')
    
    if len(dataset) > 0:
        task = dataset[0]
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Total tasks: {len(dataset)}")
        print(f"  First task: {task['task_id']}")
        print(f"  Transformation: {task['transformation']}")
        print(f"  Training examples: {task['train_inputs'].shape[0]}")
        print(f"  Grid size: {task['train_inputs'].shape[1:]}")
        
        print(f"\n  Sample input (top-left 8x8):")
        print(task['train_inputs'][0][:8, :8].numpy())
        print(f"\n  Sample output (top-left 8x8):")
        print(task['train_outputs'][0][:8, :8].numpy())
    
    print("\n" + "=" * 60)
    print("Ready to train spatial LPN!")
    print("Run: python train_spatial.py")
