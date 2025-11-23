"""
Utilities for parsing JSON responses from LLM models.
Handles markdown code fences and extra commentary.
"""
import json
from typing import Any


def extract_json_from_llm_response(text: str) -> dict:
    """
    Extract and parse JSON from an LLM response, handling:
    - Markdown code fences (```json ... ```)
    - Extra commentary before/after the JSON
    - Whitespace and formatting variations

    Args:
        text: Raw response text from LLM

    Returns:
        Parsed JSON as dict

    Raises:
        ValueError: If no valid JSON can be extracted
        json.JSONDecodeError: If JSON is malformed
    """
    cleaned = (text or "").strip()

    # Strip markdown code fences (both leading and trailing)
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    # Try to parse the whole cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: Extract JSON by finding outermost braces
        # This handles cases where there's extra text before/after the JSON
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                pass  # Fall through to raise

        # If we still can't parse, raise with helpful message
        raise ValueError(
            f"Failed to parse JSON from LLM response. "
            f"Response length: {len(text)} chars, "
            f"Preview: {text[:200]}..."
        )
