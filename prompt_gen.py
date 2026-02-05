"""
LLM-based prompt generation for image generation.

Uses LM Studio (or compatible API) to generate creative prompts.
"""

import re
import time

from openai import OpenAI

from pipeline_common import load_config, normalize_lm_studio_url, setup_logger

CONFIG = load_config()

LM_STUDIO_URL = normalize_lm_studio_url(CONFIG.get("lm_studio_url", "http://localhost:1234/v1"))
MODEL_ID = CONFIG.get("lm_studio_model", "gpt-oss-20b")

logger = setup_logger("prompt_gen", level=CONFIG.get("logging_level", "INFO"))


def generate_prompt(steering_concept=None, image_base64=None, return_details=False):
    """
    Generate a creative image generation prompt using LLM.

    Args:
        steering_concept: Optional concept to guide the prompt
        image_base64: Optional base64 image for vision-based prompt generation
        return_details: If True, return dict with metadata; else return prompt string

    Returns:
        If return_details: Dict with prompt, elapsed, model, url, mode, source, error
        Else: Prompt string

    Raises:
        RuntimeError: If LLM returns empty prompt
    """
    start_time = time.time()
    result = {
        "prompt": None,
        "elapsed": None,
        "model": MODEL_ID,
        "url": LM_STUDIO_URL,
        "mode": "vision" if image_base64 else "text",
        "source": "llm",
        "error": None
    }

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=450.0)

    if image_base64:
        if steering_concept:
            user_msg = (
                f"Look at this image and create a LONG, detailed image generation prompt (at least 80-120 words) to transform it "
                f"based on this concept: '{steering_concept}'. "
                "Describe the key elements you see, environment, mood, lighting, colors, textures, artistic style, and quality tags. "
                "Return ONLY the prompt string."
            )
        else:
            user_msg = (
                "Look at this image and create a LONG, detailed image generation prompt (at least 80-120 words) that describes it "
                "with artistic enhancements. Include subject details, environment, mood, lighting, colors, textures, artistic style, and quality tags. "
                "Return ONLY the prompt string."
            )

        if not image_base64.startswith("data:"):
            image_url = f"data:image/png;base64,{image_base64}"
        else:
            image_url = image_base64

        messages = [
            {"role": "system", "content": "You are a prompt engineer. Output ONLY the image generation prompt itself - no thinking, no reasoning, no preamble, no explanation. Start directly with the description."},
            {"role": "user", "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ]
        logger.info(f"Requesting vision-based prompt", extra={'model': MODEL_ID})
    else:
        if steering_concept:
            user_msg = (
                f"Create a LONG, detailed image prompt (at least 80-120 words) based on the concept: '{steering_concept}'. "
                "Include subject details, environment, mood, lighting, colors, textures, artistic style, and quality tags."
            )
        else:
            user_msg = (
                "Generate a LONG, detailed, creative image prompt (at least 80-120 words). "
                "Choose a unique subject and describe it with environment, mood, lighting, colors, textures, artistic style, and quality tags."
            )

        messages = [
            {"role": "system", "content": "You are a prompt engineer. Output ONLY the image generation prompt itself - no thinking, no reasoning, no preamble, no explanation. Start directly with the description."},
            {"role": "user", "content": user_msg}
        ]
        logger.info(f"Requesting text-based prompt", extra={'model': MODEL_ID})

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        msg = response.choices[0].message
        prompt = msg.content.strip() if msg.content else ""

        if not prompt and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            prompt = msg.reasoning_content.strip()
        elif not prompt and 'reasoning' in msg.model_extra:
            prompt = msg.model_extra['reasoning'].strip()

        prompt = prompt.strip('"').strip("'")

        prompt = re.sub(r'^(Got it|Okay|Alright|Let me|I\'ll|I need to|Here\'s|Here is)[^.]*\.\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'^(The user wants|This prompt)[^.]*\.\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'<think>.*?</think>', '', prompt, flags=re.DOTALL)
        prompt = re.sub(r'<reasoning>.*?</reasoning>', '', prompt, flags=re.DOTALL)
        prompt = re.sub(r'<\|begin_of_box\|>', '', prompt)
        prompt = re.sub(r'<\|end_of_box\|>', '', prompt)
        prompt = prompt.strip()

        if not prompt:
            raise RuntimeError("LLM returned empty prompt")

        result["prompt"] = prompt
        result["elapsed"] = round(time.time() - start_time, 2)
        logger.info(f"Generated prompt", extra={
            'elapsed': result['elapsed'],
            'prompt_preview': prompt[:80] + '...',
        })
        return result if return_details else prompt

    except Exception as e:
        logger.error(f"LLM generation failed", extra={'error': str(e)})
        raise


if __name__ == "__main__":
    print(generate_prompt())
