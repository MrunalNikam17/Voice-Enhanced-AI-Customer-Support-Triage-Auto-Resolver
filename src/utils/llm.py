
"""
src/utils/llm.py

Local LLM wrapper using Mistral-7B-Instruct (4-bit quantized) via
HuggingFace transformers. Loads once (singleton) and reused across all
agents (Responder, Priority, Escalation) that need LLM reasoning.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.append(os.getcwd())
from src.utils.config import config

_MODEL_NAME = config.LLM_MODEL_NAME

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading {_MODEL_NAME} (4-bit)... this takes a minute on first call.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("Model loaded.")


def local_llm_call(prompt: str, max_new_tokens: int = 300) -> str:
    """
    Sends a prompt to the local Mistral model and returns the raw generated
    text. Used as the llm_call_fn for ResponderAgent / PriorityAgent /
    EscalationAgent.
    """
    _load_model()

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(_model.device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():

        output_ids = _model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=config.LLM_TEMPERATURE,
            do_sample=config.LLM_TEMPERATURE > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_length:]

    text = _tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return text.strip()


def extract_json(text: str) -> str:
    """
    Mistral sometimes wraps JSON in extra text/markdown fences despite
    instructions. This pulls out the first {...} block so json.loads()
    doesn't choke on it.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start:end + 1]


if __name__ == "__main__":
    test_prompt = 'Respond ONLY with JSON: {"greeting": "<a short hello>"}'
    print(local_llm_call(test_prompt))