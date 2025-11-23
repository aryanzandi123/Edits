"""Shared data structures for configuring the LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class StepConfig:
    """Configuration for a single pipeline step.

    Attributes
    ----------
    name:
        Human-friendly identifier for the step. Must be unique across the pipeline.
    model:
        OpenAI model ID to use for this step.
    prompt_template:
        Primary instructions given to the model. The runner will append CSV handling
        guidelines and the prior dataset automatically.
    expected_columns:
        Ordered sequence of column names that this step must output in the CSV.
    deep_research:
        If True, adds heavyweight research instructions to the system message.
    system_prompt:
        Optional override for the system prompt. When provided, deep_research is ignored.
    reasoning_effort:
        Optional reasoning effort level (e.g., "medium", "high") passed to the
        Responses API when supported by the target model.
    use_google_search:
        Whether to enable the Google Search grounding tool for this step.
    thinking_budget:
        Optional maximum hidden reasoning tokens to request from the model.
    max_output_tokens:
        Upper bound on response tokens emitted by the model.
    search_dynamic_mode:
        When True, enables dynamic Google Search retrieval mode for aggressive research.
    search_dynamic_threshold:
        Optional stopping threshold for dynamic retrieval (lower values favor more searches).
    """

    name: str
    model: str
    prompt_template: str
    expected_columns: Sequence[str]
    deep_research: bool = False
    system_prompt: Optional[str] = None
    reasoning_effort: Optional[str] = "high"
    use_google_search: bool = True
    thinking_budget: Optional[int] = None
    max_output_tokens: Optional[int] = None
    search_dynamic_mode: bool = False
    search_dynamic_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StepConfig.name cannot be empty")
        if not self.model:
            raise ValueError(f"Step '{self.name}' must specify a model")
        if not self.prompt_template.strip():
            raise ValueError(f"Step '{self.name}' must include a prompt_template")
        if not list(self.expected_columns):
            raise ValueError(f"Step '{self.name}' must define expected_columns")


def as_columns(columns: Iterable[str]) -> list[str]:
    """Return a list of stripped column names for convenience."""

    return [col.strip() for col in columns]
