"""
Gemini Client Helper.

Single place to configure and access the Google Gemini client, so every
agent module imports from here instead of duplicating client setup.

Uses gemini-2.5-flash by default -- fast and free-tier eligible, more than
capable for this project's short, structured tasks (query parsing, strategy
selection, insight generation).

Uses the current `google-genai` SDK (the old `google-generativeai` package
was deprecated by Google and no longer receives updates or bug fixes).
"""
import os
from google import genai
from google.genai import types

_client: "genai.Client | None" = None

DEFAULT_MODEL = "gemini-3.6-flash"


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it or add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    max_output_tokens: int = 4096,
    model_name: str = DEFAULT_MODEL,
    thinking_level: str = "LOW",
) -> str:
    """
    Convenience wrapper: one prompt in, plain text out.

    thinking_level="LOW" is the key latency lever for this project: every
    call here is a short, structured task (query parsing, strategy
    selection, insight generation from already-computed evidence) -- none of
    it needs deep multi-step reasoning. Gemini 3.x's "thinking" tokens are
    real generation time (the model produces and discards them before the
    visible answer), so constraining thinking is what actually reduces
    wall-clock latency, not just the max_output_tokens ceiling.

    Note: for Gemini 3.x, the correct parameter is `thinking_level`
    (MINIMAL/LOW/MEDIUM/HIGH), not `thinking_budget` -- the older
    token-count-based parameter used by Gemini 2.5 models. Passing
    thinking_budget to a 3.x model returns 400 INVALID_ARGUMENT.
    """
    client = _get_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        ),
    )
    return (response.text or "").strip()
