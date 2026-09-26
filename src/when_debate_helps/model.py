from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 1
    max_input_tokens: int | None = None


class TransformersGenerator:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TransformersGenerator:
        name = config["name_or_path"]
        dtype_name = str(config.get("dtype", "bfloat16"))
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        device = torch.device(device_name)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=bool(config.get("trust_remote_code", False)))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        }
        if config.get("attn_implementation"):
            model_kwargs["attn_implementation"] = config["attn_implementation"]
        model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs).to(device)
        model.requires_grad_(False)
        model.generation_config.top_k = None
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=device)

    @torch.no_grad()
    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        generation: GenerationConfig,
        seed: int,
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(conversations), generation.batch_size):
            batch = conversations[start : start + generation.batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in batch
            ]
            tokenized = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=generation.max_input_tokens is not None,
                max_length=generation.max_input_tokens,
            ).to(self.device)
            batch_seed = seed + start
            random.seed(batch_seed)
            np.random.seed(batch_seed % (2**32 - 1))
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            do_sample = generation.temperature > 0
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=generation.max_new_tokens,
                do_sample=do_sample,
                temperature=generation.temperature if do_sample else None,
                top_p=generation.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            input_width = tokenized["input_ids"].shape[1]
            outputs.extend(self.tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True))
            del tokenized, generated
        return outputs

    def clear_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generation_config(config: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(config.get("max_new_tokens", 1024)),
        temperature=float(config.get("temperature", 0.0)),
        top_p=float(config.get("top_p", 1.0)),
        batch_size=int(config.get("batch_size", 1)),
        max_input_tokens=int(config["max_input_tokens"]) if config.get("max_input_tokens") else None,
    )
