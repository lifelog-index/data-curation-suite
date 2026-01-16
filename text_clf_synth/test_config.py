#!/usr/bin/env python3
"""Simple test to verify config loading and validation."""

import sys
from pathlib import Path

# Add text-clf-synth to path
sys.path.insert(0, str(Path(__file__).parent))

from config_schema import RootConfig
import yaml


def test_ielts_config():
    """Test loading IELTS config."""
    config_path = Path(__file__).parent / "examples" / "ielts_task2.yaml"
    
    print("Testing config loading...")
    print(f"Config file: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print("\n✓ YAML loaded successfully")
    
    # Validate with Pydantic
    config = RootConfig(**data)
    
    print("✓ Config validated successfully")
    print(f"\nDataset: {config.dataset.name}")
    print(f"Samples: {config.dataset.num_samples}")
    print(f"Train/Test: {config.dataset.train_test_split}")
    print(f"Stratify by: {config.dataset.stratify_by}")
    
    print(f"\nFields ({len(config.fields)}):")
    for field in config.fields:
        print(f"  - {field.name} ({field.type.value})")
        if field.type.value == "categorical":
            print(f"    Options: {field.options}")
        elif field.type.value == "numeric":
            print(f"    Range: {field.range}, Step: {field.step}")
    
    print(f"\nModel: {config.model.name}")
    print(f"Temperature: {config.model.temperature}")
    
    print(f"\nOutput:")
    print(f"  Train: {config.output.train_file}")
    print(f"  Test: {config.output.test_file}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_ielts_config()
