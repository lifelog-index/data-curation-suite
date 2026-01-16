# Quick Start Guide

## Generate IELTS Dataset (Example)

### 1. Navigate to examples directory

```bash
cd examples
```

### 2. Run the generation script

```bash
python generate_ielts.py
```

This will:
- Load your local Gemma 3 27B model
- Generate 50 IELTS Task 2 essays with scores
- Create `data/ielts_train.csv` (40 samples, 80%)
- Create `data/ielts_test.csv` (10 samples, 20%)
- Stratify by essay_type for balanced distribution

### 3. Check the output

```bash
# View train set
head -5 data/ielts_train.csv

# View test set
head -5 data/ielts_test.csv

# Count samples
wc -l data/ielts_train.csv data/ielts_test.csv
```

## Create Your Own Dataset

### 1. Create a YAML config

```yaml
# my_dataset.yaml
dataset:
  name: "my_custom_dataset"
  num_samples: 20  # Start small
  train_test_split: 0.8
  stratify_by: null

fields:
  - name: "input"
    type: "text"
    description: "Input text to classify"
  
  - name: "label"
    type: "categorical"
    options: ["class_a", "class_b", "class_c"]
    description: "The classification label"

model:
  name: "llm-checkpoints/gemma-3-27b-it-qat-autoawq"
  temperature: 0.8
  max_tokens: 2048
  tensor_parallel_size: 1
  quantization: "awq"

output:
  train_file: "my_train.csv"
  test_file: "my_test.csv"
```

### 2. Create a generation script

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from text_clf_synth import DatasetGenerator

generator = DatasetGenerator("my_dataset.yaml")
generator.generate(batch_size=1)
```

### 3. Run it

```bash
python my_script.py
```

## Tips

- **Start small**: Test with 10-20 samples first
- **Check quality**: Review a few samples before scaling up
- **Adjust temperature**: Lower (0.3-0.5) for consistency, higher (0.8-0.9) for creativity
- **Batch size**: Use 1 for reasoning tasks, 3-5 for simple classification
- **Stratification**: Enable for balanced label distribution in train/test

## Troubleshooting

### Out of Memory
- Reduce `max_tokens`
- Use smaller `batch_size`
- The model is already quantized (AWQ)

### Poor Quality
- Lower `temperature` for more consistency
- Make field descriptions more specific
- Add a `reasoning` field to guide the model

### JSON Parsing Errors
- Some samples may fail to parse (logged as warnings)
- Failed samples are skipped automatically
- Lower `temperature` to reduce parsing errors
