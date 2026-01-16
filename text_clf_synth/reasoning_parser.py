"""Parser for extracting structured data from LLM outputs."""

import json
import re
from typing import Dict, Any, List, Optional


class ReasoningParser:
    """Parse structured data from LLM reasoning outputs."""
    
    def __init__(self, fields_config: List[Any]):
        """Initialize parser with field configuration.
        
        Args:
            fields_config: List of FieldConfig objects
        """
        self.fields_config = fields_config
        self.field_names = [f.name for f in fields_config]
        self.field_types = {f.name: f.type.value for f in fields_config}
    
    def parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from model output.
        
        Args:
            text: Model output text
        
        Returns:
            Parsed JSON dict or None if parsing failed
        """
        # Try to extract JSON from text
        json_match = self._extract_json(text)
        if not json_match:
            print(f"Warning: Could not extract JSON from output")
            return None
        
        try:
            data = json.loads(json_match)
            
            # Validate and clean data
            cleaned = self._validate_and_clean(data)
            return cleaned
        
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error: {e}")
            return None
    
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text.
        
        Args:
            text: Text containing JSON
        
        Returns:
            JSON string or None
        """
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        # Look for outermost { }
        stack = []
        start_idx = None
        
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        # Found complete JSON object
                        return text[start_idx:i+1]
        
        return None
    
    
    def _validate_and_clean(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and clean parsed data.
        
        Args:
            data: Parsed JSON dict
        
        Returns:
            Cleaned dict or None if validation fails
        """
        if not isinstance(data, dict):
            print(f"Warning: Expected dict, got {type(data)}")
            return None
        
        cleaned = {}
        
        for field_name in self.field_names:
            if field_name not in data:
                print(f"Warning: Missing required field '{field_name}'")
                return None
            
            value = data[field_name]
            field_type = self.field_types[field_name]
            
            # Type validation and conversion
            if field_type == "numeric":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    print(f"Warning: Field '{field_name}' should be numeric, got '{value}'")
                    return None
            elif field_type in ["text", "categorical", "reasoning"]:
                value = str(value)
            
            cleaned[field_name] = value
        
        return cleaned
