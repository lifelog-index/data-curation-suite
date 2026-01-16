# Text-CLF-Synth: Synthetic Text Classification Dataset Generator

Generate high-quality synthetic text classification datasets using vLLM reasoning models and YAML-based configuration.

## Overview

**text-clf-synth** automates the creation of labeled text datasets by:

1. **Reading YAML configs** that define your dataset schema (fields, types, ranges, labels)
2. **Using reasoning models** (via vLLM) to generate realistic, coherent data with explanations
3. **Parsing structured output** to extract field values
4. **Writing CSV files** with automatic train/test splitting (with optional stratification)

## Features

- **YAML-based configuration** - Define your dataset schema declaratively
- **Multiple field types** - Categorical, numeric, text, and reasoning fields
- **vLLM integration** - Fast, efficient inference with local or HuggingFace models
- **Reasoning support** - Generate data with explanations for better quality
- **Batch processing** - Generate multiple samples per LLM call for efficiency

## Installation

The package is part of the data-curation-suite workspace. Install dependencies:

```bash
# From the main workspace directory
uv sync

# Add PyYAML if not already present
uv pip install pyyaml
```

## Quick Start

### 1. Create a YAML Configuration

Define your dataset schema in a YAML file:

```yaml
dataset:
  name: "my_dataset"
  num_samples: 100
  train_test_split: 0.8
  stratify_by: null  # Optional: field name for stratified split

fields:
  - name: "input_text"
    type: "text"
    description: "Description of what this field should contain"
  
  - name: "category"
    type: "categorical"
    options: ["class_a", "class_b", "class_c"]
    description: "The classification label"
  
  - name: "confidence"
    type: "numeric"
    range: [0.0, 1.0]
    step: 0.1
    description: "Confidence score"

model:
  name: "/path/to/your/model"
  temperature: 0.8
  max_tokens: 4096
  tensor_parallel_size: 1
  quantization: "awq"

output:
  train_file: "data/train.csv"
  test_file: "data/test.csv"
```

### 2. Run Generation

```python
from text_clf_synth import DatasetGenerator

# Initialize with your config
generator = DatasetGenerator("config.yaml")

# Generate the dataset
generator.generate(batch_size=1)
```

Or use it as a command-line script:

```python
#!/usr/bin/env python3
import sys
from text_clf_synth import DatasetGenerator

generator = DatasetGenerator(sys.argv[1])
generator.generate()
```

## Field Types

### Text Fields
Free-form text generation for inputs, outputs, or explanations.

```yaml
- name: "essay"
  type: "text"
  description: "A 250-word essay on the given topic"
```

### Categorical Fields
Select from predefined options.

```yaml
- name: "sentiment"
  type: "categorical"
  options: ["positive", "negative", "neutral"]
  description: "Sentiment classification"
```

### Numeric Fields
Numbers within a specified range with optional step size.

```yaml
- name: "score"
  type: "numeric"
  range: [1.0, 10.0]
  step: 0.5
  description: "Quality score from 1 to 10"
```

### Reasoning Fields
Explanations or justifications (text that explains other fields).

```yaml
- name: "reasoning"
  type: "reasoning"
  description: "Explain why the score was assigned"
```

## Example: IELTS Task 2 Dataset

See `examples/ielts_task2.yaml` for a complete example that generates:
- **Topic**: IELTS essay questions
- **Essay Type**: Opinion, discussion, problem-solution, etc.
- **Essay**: Full 250+ word essays
- **Score**: Band scores (4.0 - 9.0)
- **Reasoning**: Score justifications

Run the example:

```bash
cd text-clf-synth/examples
python generate_ielts.py
```

This will create:
- `data/ielts_train.csv` - 80% of samples (40 samples)
- `data/ielts_test.csv` - 20% of samples (10 samples)

With stratified splitting by `essay_type` to maintain label distribution.

## Configuration Reference

### Dataset Section

```yaml
dataset:
  name: str                    # Dataset name (for logging)
  num_samples: int             # Total samples to generate
  train_test_split: float      # Training ratio (0.0-1.0)
  stratify_by: str | null      # Field to stratify split (must be categorical)
```

### Model Section

```yaml
model:
  name: str                    # Model path or HuggingFace name
  temperature: float           # Sampling temperature (0.0-2.0)
  max_tokens: int              # Max tokens per generation
  tensor_parallel_size: int    # Number of GPUs
  quantization: str | null     # "awq", "gptq", or null
```

### Output Section

```yaml
output:
  train_file: str              # Path to training CSV
  test_file: str               # Path to test CSV
```

## Model Recommendations

### For Reasoning Tasks (Recommended)
- **Gemma 3 27B IT** (local) - Good balance of quality and speed
- **QwQ-32B-Preview** - Excellent reasoning capabilities
- **DeepSeek-R1** - Strong reasoning with explanations

### For Simple Classification
- **Gemma 2 9B** - Fast, efficient
- **Qwen 2.5 14B** - Good quality/speed tradeoff

## Advanced Usage

### Custom Batch Size

```python
generator.generate(batch_size=5)  # Generate 5 samples per call
```

Larger batch sizes are faster but may reduce quality. Use `batch_size=1` for reasoning tasks.

### Stratified Splitting

Ensure balanced label distribution in train/test sets:

```yaml
dataset:
  stratify_by: "label_field_name"
```

The specified field must be categorical.

## Output Format

Generated CSV files contain all fields defined in the config:

```csv
topic,essay_type,essay,score,reasoning
"Some people think...",opinion,"In today's world...",7.5,"Good coherence..."
"Discuss both views...",discussion,"There are two...",6.5,"Limited vocabulary..."
```

## Tips for High-Quality Generation

1. **Use descriptive field descriptions** - The LLM uses these to understand what to generate
2. **Start with small num_samples** - Test your config with 10-20 samples first
3. **Use reasoning fields** - Asking the model to explain improves consistency
4. **Set appropriate temperature** - 0.7-0.9 for creative tasks, 0.3-0.5 for factual data
5. **Monitor generation** - Check sample outputs during generation for quality

## Troubleshooting

### JSON Parsing Errors
- The model may occasionally produce malformed JSON
- Failed samples are logged and skipped
- Consider reducing `temperature` for more reliable output

### Out of Memory
- Reduce `max_tokens`
- Use smaller `batch_size`
- Enable quantization
- Increase `tensor_parallel_size` if you have multiple GPUs

### Poor Quality Samples
- Improve field descriptions to be more specific
- Add reasoning fields to guide the model
- Use a larger/better model
- Adjust temperature (lower for consistency, higher for creativity)

## Project Structure

```
text-clf-synth/
├── __init__.py           # Package exports
├── config_schema.py      # Pydantic models for config validation
├── generator.py          # Main orchestrator
├── vllm_client.py        # vLLM wrapper
├── reasoning_parser.py   # JSON extraction from LLM output
├── csv_writer.py         # CSV writing with train/test split
├── prompts.py            # Prompt templates
├── examples/
│   ├── ielts_task2.yaml  # IELTS example config
│   └── generate_ielts.py # IELTS generation script
└── README.md             # This file
```

## License

Part of the data-curation-suite project.
