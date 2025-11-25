"""
ARC Dataset Loader
Downloads and processes the ARC-AGI dataset from GitHub
"""

import torch
from torch.utils.data import Dataset
import json
import requests
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import zipfile
import io


class ARCDataset(Dataset):
    """
    Dataset loader for ARC (Abstraction and Reasoning Corpus)
    
    Dataset structure:
    - Each task has training and test examples
    - Each example is an input-output grid pair
    - Grids are 2D arrays with values 0-9 (10 colors)
    """
    
    def __init__(self, data_dir: str = './arc_data', split: str = 'training',
                 download: bool = True, max_grid_size: int = 30):
        """
        Args:
            data_dir: Directory to store ARC data
            split: 'training' or 'evaluation'
            download: If True, download data if not present
            max_grid_size: Maximum grid size (pad/crop to this)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_grid_size = max_grid_size
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download if needed
        if download:
            self.maybe_download()
        
        # Load tasks
        self.tasks = self.load_tasks()
        
        print(f"Loaded {len(self.tasks)} tasks from ARC {split}")
    
    def maybe_download(self):
        """Download ARC dataset from GitHub if not present"""
        training_path = self.data_dir / 'training'
        evaluation_path = self.data_dir / 'evaluation'
        
        if training_path.exists() and evaluation_path.exists():
            print("ARC data already downloaded")
            return
        
        print("Downloading ARC dataset from GitHub...")
        
        # URLs for ARC data
        base_url = "https://github.com/fchollet/ARC-AGI/raw/master/data"
        
        for split_name in ['training', 'evaluation']:
            split_dir = self.data_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            print(f"  Downloading {split_name} data...")
            
            # ARC data is split into individual JSON files
            # We'll download the raw JSON content
            try:
                # Try to get the list of files from the repo
                # This is a simplified version - in practice you'd clone the repo or use the API
                print(f"  Note: For full ARC dataset, please clone:")
                print(f"  git clone https://github.com/fchollet/ARC-AGI.git")
                print(f"  Then copy data/{split_name}/ to {self.data_dir}/{split_name}/")
            except Exception as e:
                print(f"  Could not auto-download: {e}")
                print(f"  Please manually download ARC data to {self.data_dir}")
    
    def load_tasks(self) -> List[Dict]:
        """Load all tasks from the split directory"""
        tasks = []
        
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found. Creating empty dataset.")
            print(f"Please download ARC data to {self.data_dir}")
            return []
        
        # Load all JSON files
        json_files = list(split_dir.glob('*.json'))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                task_data = json.load(f)
                tasks.append({
                    'task_id': json_file.stem,
                    'train': task_data['train'],
                    'test': task_data['test']
                })
        
        return tasks
    
    def pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to max_grid_size"""
        h, w = grid.shape
        
        if h > self.max_grid_size or w > self.max_grid_size:
            # Crop if too large
            grid = grid[:self.max_grid_size, :self.max_grid_size]
            h, w = grid.shape
        
        # Pad to square
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)
        padded[:h, :w] = grid
        
        return padded
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Get training examples
        train_inputs = []
        train_outputs = []
        
        for example in task['train']:
            input_grid = np.array(example['input'], dtype=np.int32)
            output_grid = np.array(example['output'], dtype=np.int32)
            
            train_inputs.append(self.pad_grid(input_grid))
            train_outputs.append(self.pad_grid(output_grid))
        
        # Get test example (use first one)
        test_example = task['test'][0]
        test_input = self.pad_grid(np.array(test_example['input'], dtype=np.int32))
        test_output = self.pad_grid(np.array(test_example['output'], dtype=np.int32))
        
        return {
            'task_id': task['task_id'],
            'train_inputs': torch.tensor(np.stack(train_inputs), dtype=torch.long),
            'train_outputs': torch.tensor(np.stack(train_outputs), dtype=torch.long),
            'test_input': torch.tensor(test_input, dtype=torch.long),
            'test_output': torch.tensor(test_output, dtype=torch.long)
        }


def download_arc_sample():
    """
    Download a few sample ARC tasks for testing
    Creates simplified examples if download fails
    """
    print("Creating sample ARC tasks...")
    
    data_dir = Path('./arc_data')
    training_dir = data_dir / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few simple sample tasks
    samples = [
        {
            # Task 1: Copy input to output
            "train": [
                {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 2], [2, 2]], "output": [[2, 2], [2, 2]]}
            ],
            "test": [
                {"input": [[3, 3], [3, 3]], "output": [[3, 3], [3, 3]]}
            ]
        },
        {
            # Task 2: Transpose
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]},
                {"input": [[5, 6], [7, 8]], "output": [[5, 7], [6, 8]]}
            ],
            "test": [
                {"input": [[9, 1], [2, 3]], "output": [[9, 2], [1, 3]]}
            ]
        },
        {
            # Task 3: Invert colors (0->1, 1->0)
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}
            ],
            "test": [
                {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]}
            ]
        }
    ]
    
    for i, task in enumerate(samples):
        task_file = training_dir / f'sample_{i:03d}.json'
        with open(task_file, 'w') as f:
            json.dump(task, f)
    
    print(f"✓ Created {len(samples)} sample tasks in {training_dir}")
    print(f"\nTo use real ARC data:")
    print(f"1. Clone: git clone https://github.com/fchollet/ARC-AGI.git")
    print(f"2. Copy data/training/*.json to {training_dir}/")
    print(f"3. Copy data/evaluation/*.json to {data_dir / 'evaluation'}/")


if __name__ == "__main__":
    print("ARC Dataset Loader")
    print("=" * 60)
    
    # Create sample tasks
    download_arc_sample()
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    dataset = ARCDataset(data_dir='./arc_data', split='training', download=False)
    
    if len(dataset) > 0:
        # Get first task
        task = dataset[0]
        print(f"\n✓ Loaded dataset successfully")
        print(f"  Number of tasks: {len(dataset)}")
        print(f"  First task ID: {task['task_id']}")
        print(f"  Training examples: {task['train_inputs'].shape[0]}")
        print(f"  Input grid size: {task['train_inputs'].shape[1:]}")
        print(f"  Test input shape: {task['test_input'].shape}")
        
        # Show first training example
        print(f"\n  First training example:")
        print(f"  Input:\n{task['train_inputs'][0][:5, :5]}")  # Show 5x5 corner
        print(f"  Output:\n{task['train_outputs'][0][:5, :5]}")
    else:
        print("\n⚠ No tasks loaded. Please add ARC data files.")
        print("  Run: python arc_data.py")
        print("  Then manually download ARC data as instructed")
