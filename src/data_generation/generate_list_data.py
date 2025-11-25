"""
Data generation for List Operations LPN experiment
Generates synthetic program synthesis tasks based on list transformations
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Callable
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


class ListProgramGenerator:
    """Generates list transformation programs with examples"""
    
    def __init__(self):
        self.program_types = {
            # Mapping operations (element-wise)
            'square': self._square,
            'negate': self._negate,
            'abs': self._abs,
            'add_3': self._add_3,
            'add_5': self._add_5,
            'multiply_2': self._multiply_2,
            'multiply_3': self._multiply_3,
            'subtract_1': self._subtract_1,
            'increment': self._increment,
            'decrement': self._decrement,
            
            # Filtering operations
            'filter_positive': self._filter_positive,
            'filter_negative': self._filter_negative,
            'filter_even': self._filter_even,
            'filter_odd': self._filter_odd,
            'filter_greater_than_5': self._filter_greater_than_5,
            
            # Structural operations
            'reverse': self._reverse,
            'sort_ascending': self._sort_ascending,
            'sort_descending': self._sort_descending,
            'take_first_3': self._take_first_3,
            'take_last_3': self._take_last_3,
            'duplicate_each': self._duplicate_each,
            'remove_duplicates': self._remove_duplicates,
            
            # Reduction operations (to single value)
            'sum': self._sum,
            'max': self._max,
            'min': self._min,
            'count': self._count,
            'mean': self._mean,
            
            # Combination operations
            'cumsum': self._cumsum,
            'differences': self._differences,
            'alternating_sign': self._alternating_sign,
        }
    
    # ============= Mapping Operations =============
    def _square(self, lst: List[int]) -> List[int]:
        return [x**2 for x in lst]
    
    def _negate(self, lst: List[int]) -> List[int]:
        return [-x for x in lst]
    
    def _abs(self, lst: List[int]) -> List[int]:
        return [abs(x) for x in lst]
    
    def _add_3(self, lst: List[int]) -> List[int]:
        return [x + 3 for x in lst]
    
    def _add_5(self, lst: List[int]) -> List[int]:
        return [x + 5 for x in lst]
    
    def _multiply_2(self, lst: List[int]) -> List[int]:
        return [x * 2 for x in lst]
    
    def _multiply_3(self, lst: List[int]) -> List[int]:
        return [x * 3 for x in lst]
    
    def _subtract_1(self, lst: List[int]) -> List[int]:
        return [x - 1 for x in lst]
    
    def _increment(self, lst: List[int]) -> List[int]:
        return [x + 1 for x in lst]
    
    def _decrement(self, lst: List[int]) -> List[int]:
        return [x - 1 for x in lst]
    
    # ============= Filtering Operations =============
    def _filter_positive(self, lst: List[int]) -> List[int]:
        return [x for x in lst if x > 0]
    
    def _filter_negative(self, lst: List[int]) -> List[int]:
        return [x for x in lst if x < 0]
    
    def _filter_even(self, lst: List[int]) -> List[int]:
        return [x for x in lst if x % 2 == 0]
    
    def _filter_odd(self, lst: List[int]) -> List[int]:
        return [x for x in lst if x % 2 != 0]
    
    def _filter_greater_than_5(self, lst: List[int]) -> List[int]:
        return [x for x in lst if x > 5]
    
    # ============= Structural Operations =============
    def _reverse(self, lst: List[int]) -> List[int]:
        return lst[::-1]
    
    def _sort_ascending(self, lst: List[int]) -> List[int]:
        return sorted(lst)
    
    def _sort_descending(self, lst: List[int]) -> List[int]:
        return sorted(lst, reverse=True)
    
    def _take_first_3(self, lst: List[int]) -> List[int]:
        return lst[:3]
    
    def _take_last_3(self, lst: List[int]) -> List[int]:
        return lst[-3:]
    
    def _duplicate_each(self, lst: List[int]) -> List[int]:
        result = []
        for x in lst:
            result.extend([x, x])
        return result
    
    def _remove_duplicates(self, lst: List[int]) -> List[int]:
        seen = []
        for x in lst:
            if x not in seen:
                seen.append(x)
        return seen
    
    # ============= Reduction Operations =============
    def _sum(self, lst: List[int]) -> List[int]:
        return [sum(lst)]
    
    def _max(self, lst: List[int]) -> List[int]:
        return [max(lst)] if lst else [0]
    
    def _min(self, lst: List[int]) -> List[int]:
        return [min(lst)] if lst else [0]
    
    def _count(self, lst: List[int]) -> List[int]:
        return [len(lst)]
    
    def _mean(self, lst: List[int]) -> List[int]:
        return [int(sum(lst) / len(lst))] if lst else [0]
    
    # ============= Combination Operations =============
    def _cumsum(self, lst: List[int]) -> List[int]:
        result = []
        total = 0
        for x in lst:
            total += x
            result.append(total)
        return result
    
    def _differences(self, lst: List[int]) -> List[int]:
        if len(lst) < 2:
            return []
        return [lst[i+1] - lst[i] for i in range(len(lst)-1)]
    
    def _alternating_sign(self, lst: List[int]) -> List[int]:
        return [x if i % 2 == 0 else -x for i, x in enumerate(lst)]
    
    # ============= Generation Methods =============
    def generate_input(self, program_type: str, list_length: int = None) -> List[int]:
        """Generate a random input list suitable for the program type"""
        if list_length is None:
            list_length = random.randint(4, 8)
        
        # Different input distributions for different programs
        if 'filter' in program_type:
            # Mix of positive and negative for filtering
            return [random.randint(-10, 10) for _ in range(list_length)]
        elif program_type in ['take_first_3', 'take_last_3']:
            # Ensure list is long enough
            list_length = max(list_length, 5)
            return [random.randint(-20, 20) for _ in range(list_length)]
        elif program_type in ['sum', 'max', 'min', 'mean', 'count']:
            # Smaller numbers for reductions
            return [random.randint(-10, 10) for _ in range(list_length)]
        else:
            # General case
            return [random.randint(-15, 15) for _ in range(list_length)]
    
    def generate_task(self, program_type: str, num_examples: int = 5) -> Dict:
        """Generate one task with multiple input-output examples"""
        if program_type not in self.program_types:
            raise ValueError(f"Unknown program type: {program_type}")
        
        program_fn = self.program_types[program_type]
        
        examples = []
        for _ in range(num_examples):
            input_list = self.generate_input(program_type)
            output_list = program_fn(input_list)
            examples.append({
                'input': input_list,
                'output': output_list
            })
        
        return {
            'program_type': program_type,
            'examples': examples
        }
    
    def generate_dataset(self, 
                        tasks_per_program: int = 100,
                        examples_per_task: int = 5,
                        test_examples: int = 2) -> Dict:
        """
        Generate full dataset
        
        Args:
            tasks_per_program: Number of task instances per program type
            examples_per_task: Number of train examples per task
            test_examples: Number of test examples per task
        """
        dataset = {
            'train': [],
            'val': [],
            'test': []
        }
        
        program_types = list(self.program_types.keys())
        
        for prog_type in program_types:
            print(f"Generating {tasks_per_program} tasks for '{prog_type}'...")
            
            for task_idx in range(tasks_per_program):
                # Generate training examples
                train_task = self.generate_task(prog_type, examples_per_task)
                
                # Generate test examples
                test_task = self.generate_task(prog_type, test_examples)
                
                task_data = {
                    'task_id': f"{prog_type}_{task_idx}",
                    'program_type': prog_type,
                    'train_examples': train_task['examples'],
                    'test_examples': test_task['examples']
                }
                
                # Split into train/val/test (70/15/15)
                split_rand = random.random()
                if split_rand < 0.70:
                    dataset['train'].append(task_data)
                elif split_rand < 0.85:
                    dataset['val'].append(task_data)
                else:
                    dataset['test'].append(task_data)
        
        return dataset


def main():
    """Generate and save dataset"""
    print("=" * 60)
    print("List Operations Dataset Generator")
    print("=" * 60)
    
    generator = ListProgramGenerator()
    
    print(f"\nAvailable program types: {len(generator.program_types)}")
    print(f"Programs: {', '.join(generator.program_types.keys())}\n")
    
    # Generate dataset
    print("Generating dataset...")
    dataset = generator.generate_dataset(
        tasks_per_program=100,
        examples_per_task=5,
        test_examples=2
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Total program types: {len(generator.program_types)}")
    print(f"Train tasks: {len(dataset['train'])}")
    print(f"Val tasks: {len(dataset['val'])}")
    print(f"Test tasks: {len(dataset['test'])}")
    print(f"Total tasks: {len(dataset['train']) + len(dataset['val']) + len(dataset['test'])}")
    
    # Save dataset
    output_dir = Path(r'./data/list_ops_data')
    output_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        output_path = output_dir / f"{split}.json"
        with open(output_path, 'w') as f:
            json.dump(dataset[split], f, indent=2)
        print(f"\nSaved {split} split to: {output_path}")
    
    # Show example task
    print("\n" + "=" * 60)
    print("Example Task:")
    print("=" * 60)
    example = dataset['train'][0]
    print(f"Program Type: {example['program_type']}")
    print(f"\nTraining Examples:")
    for i, ex in enumerate(example['train_examples'][:3], 1):
        print(f"  {i}. {ex['input']} → {ex['output']}")
    print(f"\nTest Examples:")
    for i, ex in enumerate(example['test_examples'], 1):
        print(f"  {i}. {ex['input']} → {ex['output']}")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
