"""Main dataset generator orchestrator."""

import yaml
from pathlib import Path
from typing import List, Dict, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .config_schema import RootConfig
from .vllm_client import VLLMClient
from .reasoning_parser import ReasoningParser
from .csv_writer import CSVWriter
from .prompts import SYSTEM_PROMPT, build_generation_prompt


class DatasetGenerator:
    """Main class for generating synthetic datasets."""
    
    def __init__(self, config_path: str):
        """Initialize generator from config file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        print(f"Loaded config: {self.config.dataset.name}")
        print(f"  Samples: {self.config.dataset.num_samples}")
        print(f"  Fields: {[f.name for f in self.config.fields]}")
    
    def _load_config(self, config_path: str) -> RootConfig:
        """Load and validate YAML configuration.
        
        Args:
            config_path: Path to YAML file
        
        Returns:
            Validated RootConfig object
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate with Pydantic
        config = RootConfig(**data)
        return config
    
    def generate(self, batch_size: int = 5):
        """Generate the complete dataset.
        
        Args:
            batch_size: Number of samples to generate per LLM call
        """
        print("\n" + "="*60)
        print(f"Starting dataset generation: {self.config.dataset.name}")
        print("="*60 + "\n")
        
        # Initialize components
        print("Step 1: Initializing vLLM client...")
        client = VLLMClient(
            model_path=self.config.model.name,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
            tensor_parallel_size=self.config.model.tensor_parallel_size,
            quantization=self.config.model.quantization,
        )
        
        print("\nStep 2: Initializing parser...")
        parser = ReasoningParser(self.config.fields)
        
        print("\nStep 3: Generating samples...")
        samples = self._generate_samples(client, parser, batch_size)
        
        print(f"\nStep 4: Writing {len(samples)} samples to CSV...")
        field_names = [f.name for f in self.config.fields]
        writer = CSVWriter(
            train_file=self.config.output.train_file,
            test_file=self.config.output.test_file,
            field_names=field_names,
            train_ratio=self.config.dataset.train_test_split,
            stratify_field=self.config.dataset.stratify_by,
        )
        writer.write_data(samples)
        
        print("\n" + "="*60)
        print("Dataset generation complete!")
        print("="*60)
        print(f"Training data: {self.config.output.train_file}")
        print(f"Test data: {self.config.output.test_file}")
    
    def _generate_samples(
        self,
        client: VLLMClient,
        parser: ReasoningParser,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Generate all samples using parallel LLM calls.
        
        Args:
            client: vLLM client
            parser: Reasoning parser
            batch_size: Parallel batch size
        
        Returns:
            List of generated samples
        """
        num_samples = self.config.dataset.num_samples
        samples = []
        
        # Prepare all individual prompts
        all_prompts = [
            build_generation_prompt(self.config.fields, i + 1)
            for i in range(num_samples)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            
            task = progress.add_task(
                f"Generating {num_samples} samples (parallel batch size: {batch_size})...",
                total=num_samples
            )
            
            # Process prompts in batches for parallelism
            for i in range(0, num_samples, batch_size):
                batch_prompts = all_prompts[i : i + batch_size]
                
                # Generate in parallel
                batch_outputs = client.generate(batch_prompts, SYSTEM_PROMPT)
                
                # Parse results individually
                for output in batch_outputs:
                    sample = parser.parse_json_output(output)
                    if sample:
                        samples.append(sample)
                    else:
                        print(f"\nWarning: A sample failed to parse, retrying once...")
                        # Simple retry for failed sample
                        # (In a more robust system, we could re-prompt with different seed/params)
                        retry_prompt = build_generation_prompt(self.config.fields, len(samples) + 1)
                        retry_output = client.generate_single(retry_prompt, SYSTEM_PROMPT)
                        retry_sample = parser.parse_json_output(retry_output)
                        if retry_sample:
                            samples.append(retry_sample)
                
                progress.update(task, completed=len(samples))
        
        return samples
    
