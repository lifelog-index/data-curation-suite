"""CSV writer with train/test splitting."""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import random


class CSVWriter:
    """Write generated data to CSV files with train/test split."""
    
    def __init__(
        self,
        train_file: str,
        test_file: str,
        field_names: List[str],
        train_ratio: float = 0.8,
        stratify_field: Optional[str] = None,
    ):
        """Initialize CSV writer.
        
        Args:
            train_file: Path to training CSV
            test_file: Path to test CSV
            field_names: List of field names (CSV columns)
            train_ratio: Ratio of samples for training
            stratify_field: Optional field name to stratify split
        """
        self.train_file = train_file
        self.test_file = test_file
        self.field_names = field_names
        self.train_ratio = train_ratio
        self.stratify_field = stratify_field
        
        # Create output directories if needed
        Path(train_file).parent.mkdir(parents=True, exist_ok=True)
        Path(test_file).parent.mkdir(parents=True, exist_ok=True)
    
    def write_data(self, samples: List[Dict[str, Any]]):
        """Write samples to CSV files with train/test split.
        
        Args:
            samples: List of data samples (dicts)
        """
        if not samples:
            print("Warning: No samples to write")
            return
        
        # Split data
        if self.stratify_field and self.stratify_field in samples[0]:
            train_samples, test_samples = self._stratified_split(samples)
        else:
            train_samples, test_samples = self._random_split(samples)
        
        print(f"Writing {len(train_samples)} training samples to {self.train_file}")
        print(f"Writing {len(test_samples)} test samples to {self.test_file}")
        
        # Write CSVs
        self._write_csv(self.train_file, train_samples)
        self._write_csv(self.test_file, test_samples)
        
        print("CSV files written successfully!")
    
    def _random_split(
        self,
        samples: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Randomly split samples into train/test.
        
        Args:
            samples: List of data samples
        
        Returns:
            Tuple of (train_samples, test_samples)
        """
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * self.train_ratio)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]
        
        return train, test
    
    def _stratified_split(
        self,
        samples: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split samples with stratification by a categorical field.
        
        Args:
            samples: List of data samples
        
        Returns:
            Tuple of (train_samples, test_samples)
        """
        # Group by stratify field
        groups = {}
        for sample in samples:
            key = sample[self.stratify_field]
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)
        
        # Split each group
        train = []
        test = []
        
        for group_samples in groups.values():
            # Shuffle group
            shuffled = group_samples.copy()
            random.shuffle(shuffled)
            
            # Split group
            split_idx = int(len(shuffled) * self.train_ratio)
            train.extend(shuffled[:split_idx])
            test.extend(shuffled[split_idx:])
        
        # Shuffle final sets
        random.shuffle(train)
        random.shuffle(test)
        
        return train, test
    
    def _write_csv(self, filepath: str, samples: List[Dict[str, Any]]):
        """Write samples to a CSV file.
        
        Args:
            filepath: Path to CSV file
            samples: List of data samples
        """
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.field_names)
            writer.writeheader()
            
            for sample in samples:
                # Only write fields that are in field_names
                row = {k: sample.get(k, '') for k in self.field_names}
                writer.writerow(row)
