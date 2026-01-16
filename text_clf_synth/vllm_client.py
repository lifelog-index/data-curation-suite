"""vLLM client wrapper for text generation."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from vllm import LLM, SamplingParams, EngineArgs


class VLLMClient:
    """Wrapper for vLLM model inference."""
    
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ):
        """Initialize vLLM client.
        
        Args:
            model_path: Path to model or HuggingFace model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tensor_parallel_size: Number of GPUs for tensor parallelism
            quantization: Quantization method (awq, gptq, None)
            max_model_len: Maximum model context length (reduce if OOM)
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"Loading model: {model_path}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  Quantization: {quantization}")
        
        # Create EngineArgs pattern as requested
        engine_args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            dtype="auto", 
        )
        
        # Convert to dict for initialization
        engine_args_dict = asdict(engine_args)
        
        self.llm = LLM(**engine_args_dict)
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,  # Let model generate naturally
        )
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate text from prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt to prepend
        
        Returns:
            List of generated texts
        """
        # Format prompts for chat models
        formatted_prompts = []
        for prompt in prompts:
            if system_prompt:
                # Format as chat for instruction-tuned models
                formatted = self._format_chat(system_prompt, prompt)
            else:
                formatted = self._format_chat("", prompt)
            formatted_prompts.append(formatted)
        
        # Generate
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            results.append(text)
        
        return results
    
    def _format_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompts for chat models.
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
        
        Returns:
            Formatted prompt string
        """
        # Gemma 3 format (adjust if using different model)
        # https://developers.googleblog.com/en/introducing-gemma3/
        formatted = f"""<bos><start_of_turn>user
{system_prompt}

{user_prompt}<end_of_turn>
<start_of_turn>model
"""
        return formatted
    
    def generate_single(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from a single prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        """
        formatted = self._format_chat("", prompt)

        results = self.generate([formatted], system_prompt)
        return results[0]


if __name__ == "__main__":
    # test load & inference model

    client = VLLMClient(model_path="/home/tnguyenho/workspace/llm-checkpoints/gemma-3-27b-it-qat-autoawq")
    print(
        client.generate_single("Hello, how are you?")
    )
    print(
        client.generate(["Hello, how are you?", "Who are you?"])
    )