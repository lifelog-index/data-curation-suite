"""Prompt templates for dataset generation."""


SYSTEM_PROMPT = """You are a data generation expert. Your task is to generate realistic, high-quality synthetic data for text classification datasets.

You will be given a schema describing the fields to generate. For each sample, you must:
1. Think through the relationships between fields
2. Generate realistic, coherent values for all fields
3. Provide reasoning for your choices (when requested)
4. Output in the exact JSON format specified

Be creative and diverse in your outputs while maintaining logical consistency."""


def build_generation_prompt(fields_config: list, sample_num: int) -> str:
    """Build a prompt for generating a single dataset sample.
    
    Args:
        fields_config: List of FieldConfig objects
        sample_num: Current sample number (for variety)
    
    Returns:
        Formatted prompt string
    """
    fields_desc = []
    
    for field in fields_config:
        desc = f"- **{field.name}** ({field.type.value}): {field.description}"
        
        if field.type.value == "categorical":
            desc += f"\n  Options: {', '.join(field.options)}"
        elif field.type.value == "numeric":
            range_desc = f"[{field.range[0]}, {field.range[1]}]"
            if field.step:
                range_desc += f" (step: {field.step})"
            desc += f"\n  Range: {range_desc}"
        
        fields_desc.append(desc)
    
    fields_text = "\n".join(fields_desc)
    
    # Build JSON schema
    json_fields = []
    for field in fields_config:
        if field.type.value == "numeric":
            json_fields.append(f'  "{field.name}": <number>')
        else:
            json_fields.append(f'  "{field.name}": "<value>"')
    
    json_schema = "{\n" + ",\n".join(json_fields) + "\n}"
    
    prompt = f"""Generate sample #{sample_num} with the following fields:

{fields_text}

Think step by step about what makes a realistic, coherent sample. Then output ONLY a valid JSON object with these exact field names:

{json_schema}

Important:
- Be diverse and creative (this is sample #{sample_num}, make it different from previous samples)
- Maintain logical consistency between fields
- For text fields, generate complete, realistic content
- For numeric fields with steps, use the specified step size
- Output ONLY the JSON, no additional text or markdown formatting"""
    
    return prompt


