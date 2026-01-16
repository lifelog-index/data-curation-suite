"""text-clf-synth: Synthetic text classification dataset generation using vLLM."""

from .generator import DatasetGenerator
from .config_schema import (
    RootConfig,
    DatasetConfigSchema,
    FieldConfig,
    ModelConfig,
    OutputConfig,
    FieldType,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetGenerator",
    "RootConfig",
    "DatasetConfigSchema",
    "FieldConfig",
    "ModelConfig",
    "OutputConfig",
    "FieldType",
]
