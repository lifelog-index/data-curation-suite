from dataclasses import asdict
from PIL import Image
import uuid as _uuid
import av


from vllm import LLM, EngineArgs, SamplingParams

def make_engine_args(model_name: str):
    return EngineArgs(
        model=model_name,
        enforce_eager=True,
        max_model_len=32768,
        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]},
        # Giới hạn memory cho multimodal mỗi prompt (tùy chỉnh theo tài nguyên)
        limit_mm_per_prompt={"video": 1},
        max_num_seqs=1,
        mm_processor_kwargs={"max_pixels": 128 * 384 * 384},
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=16384,
    )

def extract_frames_pyav(video_path, max_frames=16, sample_rate=1, resize_to=(480, 270)):
    container = av.open(video_path)
    frames = []

    for i, frame in enumerate(container.decode(video=0)):
        if i % sample_rate == 0:
            img = frame.to_image()  
            if resize_to:
                img = img.resize(resize_to, Image.LANCZOS)
            frames.append(img)

    if len(frames) == 0:
        raise RuntimeError("No frames extracted.")
    
    # uniform sampling max_frames
    if len(frames) > max_frames:
        indices = [int(i * len(frames) / max_frames) for i in range(max_frames)]
        frames = [frames[i] for i in indices]

    preview_width = 4
    preview_height = (len(frames) + preview_width - 1) // preview_width
    preview_image = Image.new('RGB', (resize_to[0] * preview_width, resize_to[1] * preview_height))
    for idx, frame in enumerate(frames):
        x = (idx % preview_width) * resize_to[0]
        y = (idx // preview_width) * resize_to[1]
        preview_image.paste(frame, (x, y))
    preview_image.save("extracted_frames_preview.jpg")
    return frames

def run_video_inference(model_name: str, video_path: str, question: str):
    engine_args = make_engine_args(model_name)
    engine_args_dict = asdict(engine_args) | {"seed": 1234, "mm_processor_cache_gb": 6}

    llm = LLM(**engine_args_dict)

    # Prompt: dùng placeholder video; token chính xác (<|video_pad|>) đã được hỗ trợ trong commit Tarsier2
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Lấy frames: bạn có thể giảm max_frames hoặc tăng sample_rate để tiết kiệm bộ nhớ
    frames = extract_frames_pyav(video_path, max_frames=128, sample_rate=3, resize_to=(480, 270))
    print(f"Extracted {len(frames)} frames from video.", flush=True)

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"video": frames},
    }

    sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        # Kiểm tra cấu trúc output theo phiên bản vLLM của bạn
        text = o.outputs[0].text.strip()
        print("=== Model output ===")
        print('\n'.join([t.strip() for t in text.split('.')]))  # 
        print("====================")

if __name__ == "__main__":
    MODEL = "omni-research/Tarsier2-Recap-7b"
    VIDEO_PATH = "/home/nhtlong/activity-description-generation/data/ads1.mp4"
    QUESTION = "Describe the video content in detail."

    run_video_inference(MODEL, VIDEO_PATH, QUESTION)