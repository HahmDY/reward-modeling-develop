import os
import logging
from typing import List, Dict, Optional, Union
from contextlib import redirect_stdout, redirect_stderr

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer
from transformers import logging as hf_logging
from vllm import LLM, SamplingParams


class VLLM:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: Optional[str] = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        trust_remote_code: bool = False,
        max_model_len: Optional[int] = None,
        revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        quiet: bool = True,
    ):
        self.model_name = model_name
        self.quiet = quiet

        if self.quiet:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            hf_logging.set_verbosity_error()
            logging.getLogger("vllm").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("urllib3").setLevel(logging.ERROR)

        tok_kwargs = {}
        if hf_token:
            tok_kwargs["token"] = hf_token
        if revision:
            tok_kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=trust_remote_code,
            **tok_kwargs,
        )

        engine_kwargs = dict(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=True,
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if revision:
            engine_kwargs["revision"] = revision

        self.llm = LLLMQuietWrapper(LLM(**engine_kwargs), quiet=self.quiet)

        self.eos_token_id = self._safe_eos_id()
        self.eot_token_id = self._id_or_none("<|eot_id|>")

    def generate(
        self,
        messages: Union[List[Dict[str, str]], str],
        chat: bool = True,
        do_sample: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        n: int = 1,
    ) -> Union[str, List[str]]:
        if chat:
            prompt = self._to_chat_prompt(messages)
        else:
            if not isinstance(messages, str):
                raise TypeError("messages must be a raw prompt string when chat=False")
            prompt = messages

        sampling = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
            n=n,
            stop=stop or None,
            stop_token_ids=[tid for tid in [self.eos_token_id, self.eot_token_id] if tid is not None],
        )

        outs = self.llm.generate([prompt], sampling_params=sampling)
        texts = [o.text for o in outs[0].outputs]
        return texts[0] if n == 1 else texts

    def _to_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise TypeError("messages must be a list of dicts with keys: role, content")

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _id_or_none(self, token: str) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            return None if tid is None or tid == self.tokenizer.unk_token_id else tid
        except Exception:
            return None

    def _safe_eos_id(self) -> Optional[int]:
        try:
            return self.tokenizer.eos_token_id
        except Exception:
            return None


class LLLMQuietWrapper:
    def __init__(self, llm: LLM, quiet: bool = True):
        self._llm = llm
        self._quiet = quiet

    def generate(self, *args, **kwargs):
        if not self._quiet:
            return self._llm.generate(*args, **kwargs)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            return self._llm.generate(*args, **kwargs)


if __name__ == "__main__":
    llm = VLLM(
        model_name="Hahmdong/PRM-qwen2.5-3b-alpacafarm-sft",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        trust_remote_code=False,
        max_model_len=None,
        hf_token=os.getenv("HF_TOKEN"),
        quiet=True,
    )

    DEMO_PRINT = True
    out = llm.generate(
        chat=True,
        messages=[
            {"role": "system", "content": "Answer in Korean."},
            {"role": "user", "content": "Explain the PPO algorithm in a simple way."},
        ],
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        max_new_tokens=512,
    )
    if DEMO_PRINT:
        print(out)
