"""Configuration schema for text-clf-synth dataset generation."""

from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class FieldType(str, Enum):
    """Supported field types for dataset generation."""
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    TEXT = "text"
    REASONING = "reasoning"


class FieldConfig(BaseModel):
    """Configuration for a single dataset field."""
    name: str = Field(..., description="Field name (will be CSV column name)")
    type: FieldType = Field(..., description="Type of field")
    description: str = Field(..., description="Description for LLM to understand field purpose")
    
    # For categorical fields
    options: Optional[List[str]] = Field(None, description="List of valid categorical values")
    
    # For numeric fields
    range: Optional[List[float]] = Field(None, description="[min, max] for numeric values")
    step: Optional[float] = Field(None, description="Step size for numeric values (e.g., 0.5)")
    
    @field_validator('options')
    @classmethod
    def validate_categorical(cls, v, info):
        """Validate categorical fields have options."""
        if info.data.get('type') == FieldType.CATEGORICAL and not v:
            raise ValueError("Categorical fields must specify 'options'")
        return v
    
    @field_validator('range')
    @classmethod
    def validate_numeric(cls, v, info):
        """Validate numeric fields have range."""
        if info.data.get('type') == FieldType.NUMERIC:
            if not v or len(v) != 2:
                raise ValueError("Numeric fields must specify 'range' as [min, max]")
            if v[0] >= v[1]:
                raise ValueError("Range min must be less than max")
        return v


class ModelConfig(BaseModel):
    """Configuration for LLM model."""
    name: str = Field(..., description="Model name or path")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(4096, gt=0, description="Maximum tokens to generate")
    tensor_parallel_size: int = Field(1, gt=0, description="Number of GPUs for tensor parallelism")
    quantization: Optional[str] = Field(None, description="Quantization method (awq, gptq, etc.)")


class OutputConfig(BaseModel):
    """Configuration for output files."""
    train_file: str = Field(..., description="Path to training CSV output")
    test_file: str = Field(..., description="Path to test CSV output")


class DatasetConfigSchema(BaseModel):
    """Configuration for dataset generation."""
    name: str = Field(..., description="Dataset name")
    num_samples: int = Field(..., gt=0, description="Total number of samples to generate")
    train_test_split: float = Field(0.8, gt=0.0, lt=1.0, description="Ratio of training samples")
    stratify_by: Optional[str] = Field(None, description="Field name to stratify split (must be categorical)")


class RootConfig(BaseModel):
    """Root configuration model."""
    dataset: DatasetConfigSchema
    fields: List[FieldConfig] = Field(..., min_length=1)
    model: ModelConfig
    output: OutputConfig
    
    @field_validator('fields')
    @classmethod
    def validate_fields_unique(cls, v):
        """Ensure field names are unique."""
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("Field names must be unique")
        return v
    
    @field_validator('dataset')
    @classmethod
    def validate_stratify(cls, v, info):
        """Validate stratify field exists and is categorical."""
        if hasattr(v, 'stratify_by') and v.stratify_by:
            # Will validate later when we have fields
            pass
        return v
