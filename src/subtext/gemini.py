"""The one Gemini caller, with the project's generation config and its retry.

There were two of these -- one in `cli.py`, one in `web/app.py` -- and they had drifted:
only the CLI's turned an empty response into a `BLOCKED` reason, so the web, which is the
path a judge actually exercises, reported every provider block as "no reason given". A
blank row that cannot be explained is exactly what `variants.BLOCKED` exists to prevent.

The retry was added after running the 102-line film set end to end: two of those requests
came back `503 UNAVAILABLE` from the API. That is 2% of a run, it is not our fault, and it
reached the page as an error. It is also transient by definition, so it is worth one more
attempt -- but only for the errors that are transient. A refusal, a bad key or a malformed
request must fail immediately and loudly; retrying those would hide a real fault behind a
slower one.
"""

from __future__ import annotations

import time

#: Attempts in total, not retries after the first. Two is the whole of the win: a 503 is a
#: moment of unavailability, and a second failure means the service is down rather than busy.
ATTEMPTS = 2

#: Seconds before the second attempt. Short, because a person is waiting on the other end.
BACKOFF = 1.5

#: Transient by nature: the service is busy or briefly gone. Everything else is a fault we
#: want to see immediately -- a bad key, a malformed request, a refusal.
TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "504")


def is_transient(exc: Exception) -> bool:
    """Whether this failure is worth one more attempt."""
    text = f"{getattr(exc, 'code', '')} {getattr(exc, 'status', '')} {exc}".upper()
    return any(marker in text for marker in TRANSIENT)


def asker(model: str | None = None):
    """One-shot Gemini call: a prompt in, the text out, `BLOCKED <reason>` when there is none."""
    from google import genai
    from google.genai import types

    from .config import settings

    client = genai.Client()
    name = model or settings().gemini_model
    config = types.GenerateContentConfig(
        temperature=0.2, max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    def ask(prompt: str) -> str:
        from .variants import BLOCKED

        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=name, contents=prompt, config=config)
                break
            except Exception as exc:
                if attempt == ATTEMPTS or not is_transient(exc):
                    raise
                time.sleep(BACKOFF)

        text = response.text or ""
        if text.strip():
            return text
        # No text. Say why, so a blank line downstream can be explained rather than read
        # as the pipeline declining the reading -- which is a different event entirely.
        reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if reason is None:
            candidates = getattr(response, "candidates", None) or []
            reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        return f"{BLOCKED} {getattr(reason, 'name', reason) or 'empty response'}"

    return ask
