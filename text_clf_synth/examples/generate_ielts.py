#!/usr/bin/env python3
"""Generate IELTS Task 2 synthetic dataset."""

import sys
from pathlib import Path

# Add parent directory to path to import text_clf_synth
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
print(f"Importing libs from {str(Path(__file__).parent.parent.parent)}")

from text_clf_synth import DatasetGenerator

config_path = Path(__file__).parent / "ielts_task2.yaml"

print("IELTS Task 2 Dataset Generator", flush=True)
print("=" * 60, flush=True)

# Initialize generator
generator = DatasetGenerator(str(config_path))

# Generate dataset (batch_size=1 for better quality with reasoning)
generator.generate(batch_size=50)

