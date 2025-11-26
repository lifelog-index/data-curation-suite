from tarsier_vllm.dataset.tarsier_datamodule import InferenceDataset
from tarsier_vllm.dataset.tarsier_datamodule import init_processor

import yaml
data_config = yaml.safe_load(open('configs/tarser2_default_config.yaml', 'r'))

generate_kwargs = {
    "do_sample": True,
    "max_new_tokens": 512,
    "top_p": 1,
    "temperature": 0,
    "use_cache": True
}


dataset = InferenceDataset(
    samples=[
        'assets/BigBuckBunny.mp4',
        'assets/BigBuckBunny.mp4',
    ],
    processor = init_processor('omni-research/Tarsier2-Recap-7b', data_config),
    config=data_config
)

print(dataset[0])