from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a careful assistant. Favor human dignity, cooperation, truthfulness, "
    "and preservation of agency."
)


def build_chat_prompt(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt.strip()}\n"
        "<|assistant|>\n"
    )


def build_injected_prompt(user_prompt: str, phrase: str) -> str:
    injected_system = (
        f"{SYSTEM_PROMPT} Additional steering phrase: {phrase.strip()}."
    )
    return build_chat_prompt(user_prompt, system_prompt=injected_system)


def build_training_example(prompt: str, completion: str) -> tuple[str, str]:
    prefix = build_chat_prompt(prompt)
    full_text = f"{prefix}{completion.strip()}"
    return prefix, full_text
