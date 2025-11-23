#!/usr/bin/env python3
"""
ENHANCED Pipeline Runner with Evidence Validation
Runs the maximized pipeline + optional post-validation for citations
UPDATED: Flask-compatible with run_full_job() for web integration
UPDATED: Protein database integration for cross-query knowledge building
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
import os
import sys
import time
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

# Fix Windows console encoding for Greek letters and special characters
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import httpx
from google.genai import types, errors as genai_errors

from dotenv import load_dotenv

# Import protein database for cross-query knowledge
import utils.protein_database as pdb

# Import the MAXIMIZED config (supports dynamic rounds)
try:
    from pipeline.config_dynamic import generate_pipeline, PIPELINE_STEPS as DEFAULT_PIPELINE_STEPS
    DYNAMIC_CONFIG_AVAILABLE = True
except ImportError:
    from pipeline.config_gemini_MAXIMIZED import PIPELINE_STEPS as DEFAULT_PIPELINE_STEPS
    DYNAMIC_CONFIG_AVAILABLE = False

from pipeline.types import StepConfig
from visualizer import create_visualization

# Import evidence validator (optional)
try:
    from utils.evidence_validator import validate_and_enrich_evidence
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False

# Import claim fact checker (optional)
try:
    from utils.claim_fact_checker import fact_check_json
    FACT_CHECKER_AVAILABLE = True
except ImportError:
    FACT_CHECKER_AVAILABLE = False

# Import step logger for comprehensive logging
try:
    from utils.step_logger import StepLogger
    STEP_LOGGER_AVAILABLE = True
except ImportError:
    STEP_LOGGER_AVAILABLE = False

# Import function deduplicator (optional)
try:
    from utils.deduplicate_functions import deduplicate_payload
    DEDUPLICATOR_AVAILABLE = True
except ImportError:
    DEDUPLICATOR_AVAILABLE = False
    deduplicate_payload = None  # Define as None when not available

# Import PMID updater (optional)
try:
    from utils.update_cache_pmids import update_payload_pmids
    PMID_UPDATER_AVAILABLE = True
except ImportError:
    PMID_UPDATER_AVAILABLE = False
    update_payload_pmids = None  # Define as None when not available
    print("[WARN]update_cache_pmids not available - PMIDs will not be validated", file=sys.stderr)

# Import function name cleaner (optional)
try:
    from utils.clean_function_names import clean_payload_function_names
    FUNCTION_NAME_CLEANER_AVAILABLE = True
except ImportError:
    FUNCTION_NAME_CLEANER_AVAILABLE = False
    clean_payload_function_names = None

# Import interaction metadata generator (optional)
try:
    from utils.interaction_metadata_generator import generate_interaction_metadata
    METADATA_GENERATOR_AVAILABLE = True
except ImportError:
    METADATA_GENERATOR_AVAILABLE = False
    generate_interaction_metadata = None

# Import integrated arrow validator (includes direct link extraction)
try:
    from utils.arrow_validator_integrated import validate_arrows_for_payload
    ARROW_VALIDATOR_AVAILABLE = True
except ImportError:
    # Fallback to basic arrow validator if integrated one not available
    try:
        from utils.arrow_effect_validator import validate_arrows_and_effects as validate_arrows_for_payload
        ARROW_VALIDATOR_AVAILABLE = True
    except ImportError:
        ARROW_VALIDATOR_AVAILABLE = False
        validate_arrows_for_payload = None

# Import schema validator (for pre/post validation gates)
try:
    from utils.schema_validator import (
        validate_schema_consistency,
        finalize_interaction_metadata
    )
    SCHEMA_VALIDATOR_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATOR_AVAILABLE = False
    validate_schema_consistency = None
    finalize_interaction_metadata = None

MAX_ALLOWED_THINKING_BUDGET = 32768
MIN_ALLOWED_THINKING_BUDGET = 1000

CACHE_DIR = "cache"


def _coerce_token_count(value: Any) -> int:
    """Safely convert token counters to ints, treating None/missing as 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0



class PipelineError(RuntimeError):
    """Raised when a pipeline step fails validation or parsing."""


def ensure_env() -> None:
    """Load environment variables and verify the Google API key exists."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")


def validate_steps(steps: Iterable[StepConfig]) -> List[StepConfig]:
    """Ensure step configuration is sane before executing the pipeline."""
    seen_names: set[str] = set()
    validated: List[StepConfig] = []

    for step in steps:
        if step.name in seen_names:
            raise PipelineError(f"Duplicate step name detected: {step.name}")
        if not step.expected_columns:
            raise PipelineError(f"Step '{step.name}' must declare expected_columns.")
        seen_names.add(step.name)
        validated.append(step)

    if not validated:
        raise PipelineError("PIPELINE_STEPS is empty.")

    return validated


def strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences if present."""
    if text is None:
        return ""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
        elif stripped.lower().startswith("csv"):
            stripped = stripped[3:].lstrip()
    return stripped


def deep_merge_interactors(
    existing_interactors: List[Dict[str, Any]],
    new_interactors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Intelligently merge new interactors into existing list.

    - If interactor (by 'primary' key) doesn't exist, add it
    - If interactor exists, merge new fields and append new functions
    """
    # Create lookup by primary key
    interactor_map = {i.get("primary"): deepcopy(i) for i in existing_interactors}

    for new_int in new_interactors:
        primary_key = new_int.get("primary")
        if not primary_key:
            continue

        if primary_key in interactor_map:
            # Merge into existing interactor
            existing = interactor_map[primary_key]

            # Merge functions (update in place by signature; append new ones)
            existing_functions = existing.get("functions", []) or []
            new_functions = new_int.get("functions", []) or []

            def _norm_str(v: Any) -> str:
                return str(v or "").strip().lower()

            def _norm_dir(fn: Dict[str, Any]) -> str:
                d = _norm_str(fn.get("interaction_direction") or fn.get("direction"))
                if "bid" in d:
                    return "bidirectional"
                if "primary_to_main" in d or d == "p2m" or d == "b_to_a":
                    return "primary_to_main"
                return "main_to_primary"

            def _fn_signature(fn: Dict[str, Any]) -> str:
                if "mechanism_id" in fn and fn.get("mechanism_id"):
                    return f"id:{_norm_str(fn.get('mechanism_id'))}|dir:{_norm_dir(fn)}"
                name = _norm_str(fn.get("function"))
                proc = _norm_str(fn.get("cellular_process"))
                return f"name:{name}|proc:{proc}|dir:{_norm_dir(fn)}"

            # Build index for existing functions
            existing_index: Dict[str, Dict[str, Any]] = {}
            for ef in existing_functions:
                if isinstance(ef, dict):
                    existing_index[_fn_signature(ef)] = ef

            # Determine context type for tagging
            interaction_type = new_int.get("interaction_type", "direct")
            upstream = new_int.get("upstream_interactor")
            mediator_chain = new_int.get("mediator_chain", [])
            context_type = "chain" if (interaction_type == "indirect" or upstream or mediator_chain) else "direct"

            for nf in new_functions:
                if not isinstance(nf, dict):
                    continue
                sig = _fn_signature(nf)
                if sig in existing_index:
                    base = existing_index[sig]
                    # Update core directional/effect fields if provided
                    for k in ("arrow", "interaction_effect", "direction", "interaction_direction", "intent"):
                        v = nf.get(k)
                        if v not in (None, ""):
                            base[k] = v
                    # Merge pmids
                    base_pmids = set(base.get("pmids", []) or [])
                    new_pmids = set(nf.get("pmids", []) or [])
                    if new_pmids:
                        base["pmids"] = sorted(list(base_pmids.union(new_pmids)))
                    # Merge specific_effects
                    base_se = set(base.get("specific_effects", []) or [])
                    new_se = set(nf.get("specific_effects", []) or [])
                    if new_se:
                        base["specific_effects"] = sorted(list(base_se.union(new_se)))
                    # Merge biological_consequence
                    base_bc = set(map(str, (base.get("biological_consequence", []) or [])))
                    new_bc = set(map(str, (nf.get("biological_consequence", []) or [])))
                    if new_bc:
                        base["biological_consequence"] = sorted(list(base_bc.union(new_bc)))
                    # Merge evidence by PMID/id key
                    def _ek(e: Dict[str, Any]) -> str:
                        return str((e or {}).get("pmid") or (e or {}).get("id") or "")
                    base_ev = base.get("evidence", []) or []
                    ev_map = { _ek(e): e for e in base_ev if isinstance(e, dict) }
                    for e in (nf.get("evidence", []) or []):
                        if isinstance(e, dict):
                            k = _ek(e)
                            if k and k in ev_map:
                                if len(str(e)) > len(str(ev_map[k])):
                                    ev_map[k] = e
                            else:
                                ev_map[k] = e
                    base["evidence"] = list(ev_map.values())
                else:
                    # Tag and append as new function
                    if "_context" not in nf:
                        nf["_context"] = {
                            "type": context_type,
                            "query_protein": new_int.get("_query_protein"),
                            "chain": mediator_chain if mediator_chain else None,
                        }
                    existing_functions.append(nf)

            existing["functions"] = existing_functions

            # Update other fields (take newer values)
            for key, value in new_int.items():
                if key == "functions":
                    continue  # Already handled
                elif key == "pmids":
                    # Merge PMIDs (union)
                    existing_pmids = existing.get("pmids", [])
                    existing["pmids"] = list(set(existing_pmids + value))
                elif key == "evidence":
                    # Append new evidence
                    existing_evidence = existing.get("evidence", [])
                    existing["evidence"] = existing_evidence + value
                elif key == "interaction_type":
                    # CRITICAL: Preserve interaction_type from Phase 1 (discovery)
                    # Only update if existing doesn't have it OR new value is explicit classification
                    if not existing.get("interaction_type") and value in ["direct", "indirect"]:
                        existing[key] = value
                    # If existing has it, NEVER overwrite (Phase 1 is authoritative)
                elif key == "upstream_interactor":
                    # Preserve upstream_interactor for indirect interactors
                    if not existing.get("upstream_interactor") and value:
                        existing[key] = value
                else:
                    # Overwrite with new value
                    existing[key] = value
        else:
            # New interactor - add it
            interactor_map[primary_key] = deepcopy(new_int)

    return list(interactor_map.values())


def aggregate_function_arrows(interactor: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate function-level arrows into interaction-level arrows field.

    UPDATED: Now handles both interaction_effect (protein-level) and arrow (function-level).
    Aggregates interaction_direction and interaction_effect into interactor-level fields.

    This function computes:
    - `arrows`: Dict mapping direction → list of unique interaction_effect types
    - `arrow`: Backward-compat field (most common interaction_effect or 'complex' if mixed)
    - `direction`: main_to_primary | primary_to_main | bidirectional

    Args:
        interactor: Interactor dict with functions[] containing interaction_effect/interaction_direction fields

    Returns:
        Updated interactor dict with arrows and arrow fields
    """
    functions = interactor.get("functions", [])

    if not functions:
        # No functions: default to 'binds' main_to_primary (NOT bidirectional)
        interactor["arrow"] = "binds"
        interactor["arrows"] = {"main_to_primary": ["binds"]}
        interactor["direction"] = "main_to_primary"
        return interactor

    # Collect arrows by direction AND count function directions
    # NOTE: We now use interaction_effect instead of arrow for protein-level aggregation
    arrows_by_direction = {
        "main_to_primary": set(),
        "primary_to_main": set(),
        "bidirectional": set()
    }

    direction_counts = {
        "main_to_primary": 0,
        "primary_to_main": 0,
        "bidirectional": 0
    }

    for fn in functions:
        if not isinstance(fn, dict):
            continue

        # NEW: Use interaction_effect for protein-level aggregation
        # Fallback to arrow for backward compatibility with old data
        interaction_effect = fn.get("interaction_effect", fn.get("arrow", "complex"))

        # NEW: Use interaction_direction for per-function direction
        # Fallback to direction for backward compatibility with old data
        interaction_direction = fn.get("interaction_direction", fn.get("direction", "main_to_primary"))

        # Count this function's direction
        direction_counts[interaction_direction] += 1

        # Add interaction_effect to direction set (NO automatic cross-addition for bidirectional!)
        arrows_by_direction[interaction_direction].add(interaction_effect)

    # Convert sets to lists
    arrows = {
        k: sorted(list(v)) if v else []
        for k, v in arrows_by_direction.items()
    }

    # Remove empty directions
    arrows = {k: v for k, v in arrows.items() if v}

    # Determine summary arrow field (align with metadata generator semantics)
    all_arrows = set()
    for arrow_list in arrows.values():
        all_arrows.update(arrow_list)

    if len(all_arrows) == 0:
        arrow = "binds"  # Fallback
    elif len(all_arrows) == 1:
        arrow = list(all_arrows)[0]  # Single arrow type
    else:
        # Mixed effects -> use 'regulates' to match user-facing semantics
        arrow = "regulates"

    # Determine primary direction (FIXED LOGIC)
    total_functions = len(functions)
    bidirectional_count = direction_counts["bidirectional"]
    main_to_primary_count = direction_counts["main_to_primary"]
    primary_to_main_count = direction_counts["primary_to_main"]

    # Only mark as bidirectional if:
    # 1. Majority (>50%) of functions are explicitly bidirectional, OR
    # 2. At least 30% of functions are in EACH direction (main_to_primary AND primary_to_main)
    # Otherwise, use the dominant direction
    if bidirectional_count > total_functions / 2:
        # Majority explicitly bidirectional
        direction = "bidirectional"
    elif (main_to_primary_count >= total_functions * 0.3 and
          primary_to_main_count >= total_functions * 0.3):
        # Significant functions in BOTH directions
        direction = "bidirectional"
    elif primary_to_main_count > main_to_primary_count:
        # More primary_to_main functions
        direction = "primary_to_main"
    else:
        # Default to main_to_primary (includes ties)
        direction = "main_to_primary"

    # Update interactor
    interactor["arrows"] = arrows
    interactor["arrow"] = arrow
    interactor["direction"] = direction

    return interactor


def parse_json_output(
    text: str,
    expected_fields: List[str],
    previous_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse model output into a JSON object, merging with prior payload when needed.

    This function performs DIFFERENTIAL MERGING:
    - New interactors are added to the list
    - Existing interactors get new functions appended
    - Tracking lists (interactor_history, etc.) are appended
    """
    cleaned = strip_code_fences(text)

    # Handle None or empty input (e.g., from cancelled jobs)
    if not cleaned or not cleaned.strip():
        raise PipelineError("Empty or null model output (job may have been cancelled).")

    decoder = json.JSONDecoder()
    idx = 0
    data_segments: List[Dict[str, Any]] = []
    last_end: Optional[int] = None
    required = [field for field in expected_fields if field != "ndjson"]

    # Scan for all JSON objects in response
    while idx < len(cleaned):
        try:
            obj, end_idx = decoder.raw_decode(cleaned, idx)
            last_end = end_idx
            idx = end_idx
            while idx < len(cleaned) and cleaned[idx] in (" ", "\n", "\r", "\t"):
                idx += 1

            candidate_dicts: List[Dict[str, Any]] = []
            if isinstance(obj, dict):
                candidate_dicts.append(obj)
            elif isinstance(obj, list):
                candidate_dicts.extend(item for item in obj if isinstance(item, dict))

            data_segments.extend(candidate_dicts)

            if any(all(field in segment for field in required) for segment in candidate_dicts):
                break
        except json.JSONDecodeError:
            idx += 1

    if not data_segments:
        raise PipelineError("No valid JSON found in model output.")

    # Merge all segments from this step's output
    step_output: Dict[str, Any] = {}
    for segment in data_segments:
        if not isinstance(segment, dict):
            continue
        for key, value in segment.items():
            if key in step_output and isinstance(step_output[key], dict) and isinstance(value, dict):
                step_output[key].update(value)
            else:
                step_output[key] = value

    # DIFFERENTIAL MERGE with previous payload
    if previous_payload:
        merged: Dict[str, Any] = deepcopy(previous_payload)

        # Handle ctx_json specially (intelligent merge)
        if "ctx_json" in step_output:
            new_ctx = step_output["ctx_json"]
            existing_ctx = merged.get("ctx_json", {})

            # Always use new 'main' if provided
            if "main" in new_ctx:
                existing_ctx["main"] = new_ctx["main"]

            # Merge interactors intelligently
            if "interactors" in new_ctx:
                existing_interactors = existing_ctx.get("interactors", [])
                new_interactors = new_ctx["interactors"]
                existing_ctx["interactors"] = deep_merge_interactors(existing_interactors, new_interactors)

            # Append to tracking lists
            for list_key in ["interactor_history", "function_batches", "search_history"]:
                if list_key in new_ctx:
                    existing_list = existing_ctx.get(list_key, [])
                    new_items = new_ctx[list_key]
                    # Append unique items
                    existing_ctx[list_key] = existing_list + [x for x in new_items if x not in existing_list]

            # Merge function_history (dict of lists)
            if "function_history" in new_ctx:
                existing_func_hist = existing_ctx.get("function_history", {})
                new_func_hist = new_ctx["function_history"]
                for protein, funcs in new_func_hist.items():
                    if protein in existing_func_hist:
                        existing_func_hist[protein].extend(funcs)
                    else:
                        existing_func_hist[protein] = funcs
                existing_ctx["function_history"] = existing_func_hist

            merged["ctx_json"] = existing_ctx

        # Merge other top-level fields
        for key, value in step_output.items():
            if key == "ctx_json":
                continue  # Already handled
            merged[key] = value

        result = merged
    else:
        result = step_output

    # Validate required fields
    missing = [field for field in required if field not in result]
    if missing:
        raise PipelineError(f"Missing required fields in output: {missing}")

    return result


def call_gemini_model(step: StepConfig, prompt: str, cancel_event=None) -> tuple[str, Dict[str, int]]:
    """Execute a single model call with dynamic thinking budget.

    Args:
        step: Step configuration
        prompt: Prompt text
        cancel_event: Optional threading.Event to check for cancellation

    Returns:
        tuple: (response_text, token_stats_dict)
            token_stats_dict contains: {
                'thinking_tokens': int,
                'output_tokens': int,
                'total_tokens': int
            }

    Raises:
        PipelineError: If cancellation is requested
    """
    from google import genai as google_genai

    # Check for cancellation before making expensive API call
    if cancel_event and cancel_event.is_set():
        raise PipelineError("Job cancelled by user")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise PipelineError("GOOGLE_API_KEY not found in environment")

    client = google_genai.Client(api_key=api_key)

    def build_generation_config() -> tuple:
        config_dict: dict[str, Any] = {}
        applied_thinking_budget: Optional[int] = None

        # Dynamic thinking budget: Let model decide unless explicitly set
        requested_thinking_budget = getattr(step, "thinking_budget", None)
        if requested_thinking_budget is not None:
            # Explicit budget set - use it
            clamped_budget = max(MIN_ALLOWED_THINKING_BUDGET,
                                min(requested_thinking_budget, MAX_ALLOWED_THINKING_BUDGET))
            applied_thinking_budget = clamped_budget
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_budget=clamped_budget,
                include_thoughts=True,
            )
        # If no explicit budget, let model decide (don't set thinking_config)

        # Configure tools
        tools: list[types.Tool] = []
        if step.use_google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        if hasattr(step, "use_url_context") and step.use_url_context:
            tools.append(types.Tool(url_context=types.UrlContext()))

        if hasattr(step, "use_code_execution") and step.use_code_execution:
            tools.append(types.Tool(code_execution=types.CodeExecution()))

        if tools:
            config_dict["tools"] = tools

        # System instructions
        if step.system_prompt:
            config_dict["system_instructions"] = types.Content(
                parts=[types.Part(text=step.system_prompt)]
            )

        # Output tokens
        output_token_limit = getattr(step, "max_output_tokens", None) or 65536
        config_dict["max_output_tokens"] = output_token_limit
        config_dict["temperature"] = 0.3
        config_dict["top_p"] = 0.90

        return (types.GenerateContentConfig(**config_dict), applied_thinking_budget)

    max_retries = 8
    base_delay = 5.0

    for attempt in range(1, max_retries + 1):
        config, thinking_budget = build_generation_config()

        try:
            if attempt == 1:
                print(f"   Calling gemini-2.5-pro (thinking: {thinking_budget or 'auto'})", flush=True)

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config,
            )

            # Extract token usage statistics
            token_stats = {
                'thinking_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }

            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                # Extract thinking tokens (if available)
                if hasattr(usage, 'cached_content_token_count'):
                    token_stats['thinking_tokens'] = _coerce_token_count(getattr(usage, 'cached_content_token_count', 0))
                # More reliable: check for thoughts in response
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'grounding_metadata'):
                            # Thinking tokens might be in metadata
                            pass

                # Output tokens
                token_stats['output_tokens'] = _coerce_token_count(getattr(usage, 'candidates_token_count', 0))
                # Total tokens
                token_stats['total_tokens'] = _coerce_token_count(getattr(usage, 'total_token_count', 0))

                # If we have total but not thinking, estimate thinking
                if token_stats['total_tokens'] > 0 and token_stats['thinking_tokens'] == 0:
                    input_tokens = _coerce_token_count(getattr(usage, 'prompt_token_count', 0))
                    token_stats['thinking_tokens'] = max(0, token_stats['total_tokens'] - input_tokens - token_stats['output_tokens'])

            # Extract output text
            output_text = ''
            if hasattr(response, 'text'):
                output_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                output_text = ''.join(part.text for part in parts if hasattr(part, 'text'))
            else:
                raise PipelineError("No text in response")

            token_stats = {key: _coerce_token_count(value) for key, value in token_stats.items()}
            return output_text, token_stats

        except Exception as e:
            error_str = str(e)
            is_quota_error = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            
            if is_quota_error:
                # Aggressive backoff for quota errors (min 30s, scaling up)
                delay = 30.0 * attempt
                print(f"   [WARN]Quota exceeded (429). Pausing for {delay:.1f}s to recover...", flush=True)
            else:
                # Standard exponential backoff for other errors
                delay = base_delay * (2 ** (attempt - 1))
                
            if attempt < max_retries:
                if not is_quota_error:
                    print(f"   [WARN]Error: {e}. Retrying in {delay:.1f}s...", flush=True)
                time.sleep(delay)
            else:
                raise PipelineError(f"Failed after {max_retries} attempts: {e}")


def create_snapshot_from_ctx(
    ctx_json: Dict[str, Any],
    expected_fields: List[str],
    step_name: str,
) -> Dict[str, Any]:
    """Generate snapshot_json and ndjson from ctx_json."""
    main_symbol = ctx_json.get("main", "UNKNOWN")
    interactors_data = ctx_json.get("interactors", [])

    snapshot_interactors: List[Dict[str, Any]] = []
    ndjson_lines: List[str] = []

    for interactor in interactors_data:
        # Extract core fields
        primary = interactor.get("primary", "")
        direction = interactor.get("direction", "")
        arrow = interactor.get("arrow", "")
        intent = interactor.get("intent", "")
        pmids = interactor.get("pmids", [])
        confidence = interactor.get("confidence")
        evidence = interactor.get("evidence", [])
        support_summary = interactor.get("support_summary", "")
        multiple_mechanisms = interactor.get("multiple_mechanisms", False)

        # Minimal functions (without full evidence to reduce size)
        functions_full = interactor.get("functions", [])
        minimal_functions: List[Dict[str, Any]] = []
        for func in functions_full:
            minimal_func = {
                "function": func.get("function", ""),
                "arrow": func.get("arrow", ""),
                "interaction_effect": func.get("interaction_effect", func.get("arrow", "")),
                "interaction_direction": func.get("interaction_direction", func.get("direction", "")),
                "cellular_process": func.get("cellular_process", ""),
                "biological_consequence": func.get("biological_consequence", []),
                "specific_effects": func.get("specific_effects", []),
                "pmids": func.get("pmids", []),
                "confidence": func.get("confidence"),
            }
            if "mechanism_id" in func:
                minimal_func["mechanism_id"] = func["mechanism_id"]
            if "evidence" in func:
                minimal_func["evidence"] = func["evidence"]
            minimal_functions.append(minimal_func)

        # Build snapshot interactor entry
        interactor_entry: Dict[str, Any] = {}
        if primary:
            interactor_entry["primary"] = primary
        if direction:
            interactor_entry["direction"] = direction
        if arrow:
            interactor_entry["arrow"] = arrow
        if intent:
            interactor_entry["intent"] = intent
        if pmids:
            interactor_entry["pmids"] = pmids
        if confidence is not None:
            interactor_entry["confidence"] = confidence
        if evidence:
            interactor_entry["evidence"] = evidence
        if support_summary:
            interactor_entry["support_summary"] = support_summary
        if multiple_mechanisms:
            interactor_entry["multiple_mechanisms"] = multiple_mechanisms
        interactor_entry["functions"] = minimal_functions

        snapshot_interactors.append(interactor_entry)

        # NDJSON line
        ndjson_obj: Dict[str, Any] = {"main": main_symbol}
        ndjson_obj.update(interactor_entry)
        ndjson_lines.append(json.dumps(ndjson_obj, ensure_ascii=False, separators=(",", ":")))

    snapshot_json = {"main": main_symbol, "interactors": snapshot_interactors}
    result: Dict[str, Any] = {
        "ctx_json": ctx_json,
        "snapshot_json": snapshot_json,
        "ndjson": ndjson_lines
    }
    result["step_json"] = {"step": step_name, "rows": len(ndjson_lines)}

    for field in expected_fields:
        if field not in result:
            result[field] = None

    return result


def dumps_compact(data: Any) -> str:
    """Serialize data to compact JSON for prompts."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_known_interactions_context(known_interactions: List[Dict[str, Any]]) -> str:
    """
    Build exclusion context from known interactions to avoid re-searching.

    Args:
        known_interactions: List of interaction dicts from protein database

    Returns:
        Formatted string with known interactions to skip
    """
    if not known_interactions:
        return ""

    context = "\n\n" + "="*80 + "\n"
    context += "KNOWN INTERACTIONS DATABASE (DO NOT RE-SEARCH)\n"
    context += "="*80 + "\n\n"
    context += f"The following interactions have ALREADY been discovered in previous queries.\n"
    context += f"DO NOT search for or report these again. Focus ONLY on finding NEW interactions.\n\n"

    for i, interaction in enumerate(known_interactions[:50], 1):  # Limit to 50 to avoid token bloat
        partner = interaction.get('primary', 'Unknown')
        confidence = interaction.get('confidence', 0.0)
        context += f"  {i}. {partner} (confidence: {confidence:.2f})\n"

    if len(known_interactions) > 50:
        context += f"\n... and {len(known_interactions) - 50} more known interactions.\n"

    context += f"\nTOTAL KNOWN INTERACTIONS: {len(known_interactions)}\n"
    context += "="*80 + "\n"
    context += "**YOUR TASK: Find ONLY interactions NOT listed above.**\n"
    context += "="*80 + "\n\n"

    return context


def build_prompt(
    step: StepConfig,
    prior_payload: Optional[Dict[str, Any]],
    user_query: str,
    is_first_step: bool,
    known_interactions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build the prompt for this step.

    Args:
        step: Step configuration
        prior_payload: Previous step's output
        user_query: Protein name
        is_first_step: Whether this is the first step
        known_interactions: List of known interactions from database (for exclusion)
    """
    expected_fields = [field.strip() for field in step.expected_columns]
    instructions = [
        "Return ONLY valid JSON. No markdown fences, no extra text.",
        f"Keys required: {', '.join(expected_fields)}.",
        "Preserve existing ctx_json content; extend it with this step's additions.",
    ]

    if prior_payload and "ctx_json" in prior_payload:
        ctx_compact = dumps_compact(prior_payload["ctx_json"])
        instructions.append(f"CONTEXT (from previous steps):\n{ctx_compact}")
    else:
        instructions.append("This is the first step; initialize ctx_json.")

    full_prompt = "\n".join(instructions)

    # Substitute placeholders in template
    template = step.prompt_template
    if "{user_query}" in template:
        template = template.replace("{user_query}", user_query)
    if "{ctx_json.main}" in template and prior_payload:
        main = prior_payload.get("ctx_json", {}).get("main", user_query)
        template = template.replace("{ctx_json.main}", main)
    if "{ctx_json.interactor_history}" in template and prior_payload:
        history = prior_payload.get("ctx_json", {}).get("interactor_history", [])
        template = template.replace("{ctx_json.interactor_history}", str(history))
    if "{ctx_json.function_batches}" in template and prior_payload:
        batches = prior_payload.get("ctx_json", {}).get("function_batches", [])
        template = template.replace("{ctx_json.function_batches}", str(batches))
    if "{ctx_json.function_history}" in template and prior_payload:
        func_hist = prior_payload.get("ctx_json", {}).get("function_history", {})
        template = template.replace("{ctx_json.function_history}", dumps_compact(func_hist))

    full_prompt += "\n\n" + template

    # Add known interactions exclusion context for interactor discovery steps
    if known_interactions and ("discover" in step.name.lower() or "step1" in step.name.lower()):
        exclusion_context = build_known_interactions_context(known_interactions)
        full_prompt += exclusion_context

    return full_prompt



# ═══════════════════════════════════════════════════════════════
# VALIDATION HELPER FUNCTIONS FOR PHASE 2 COMPLETENESS
# ═══════════════════════════════════════════════════════════════

def find_interactors_without_functions(ctx_json: dict) -> list[dict]:
    """
    Find all interactors that are missing functions.
    
    Args:
        ctx_json: The current context JSON
        
    Returns:
        List of dicts with {name, interaction_type, functions_count}
    """
    missing = []
    interactors = ctx_json.get("interactors", [])
    
    for interactor in interactors:
        name = interactor.get("primary", "Unknown")
        interaction_type = interactor.get("interaction_type", "direct")  # Default to direct if missing
        functions = interactor.get("functions", [])
        
        if not functions or len(functions) == 0:
            missing.append({
                "name": name,
                "interaction_type": interaction_type,
                "functions_count": 0
            })
    
    return missing


def validate_phase2_completeness(ctx_json: dict, interactor_history: list[str]) -> tuple[bool, list[dict]]:
    """
    Validate that ALL interactors from interactor_history have functions in ctx_json.
    
    Args:
        ctx_json: The current context JSON
        interactor_history: List of all discovered interactor names
        
    Returns:
        Tuple of (is_complete: bool, missing_interactors: list[dict])
    """
    missing = find_interactors_without_functions(ctx_json)
    
    # Also check if any interactors in history are NOT in ctx_json.interactors at all
    interactors_map = {i.get("primary"): i for i in ctx_json.get("interactors", [])}
    
    for name in interactor_history:
        if name not in interactors_map:
            missing.append({
                "name": name,
                "interaction_type": "unknown",
                "functions_count": 0,
                "note": "Not found in ctx_json.interactors"
            })
    
    is_complete = len(missing) == 0
    return is_complete, missing


def log_missing_functions_diagnostic(
    ctx_json: dict,
    interactor_history: list[str],
    step_name: str = "unknown"
) -> None:
    """
    Log detailed diagnostic information about missing functions.
    
    Args:
        ctx_json: The current context JSON
        interactor_history: List of all discovered interactor names
        step_name: Name of the step where this diagnostic is run
    """
    is_complete, missing = validate_phase2_completeness(ctx_json, interactor_history)
    
    if not is_complete:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[VALIDATION] [WARN] PHASE 2 INCOMPLETE AFTER {step_name}", file=sys.stderr)
        print(f"[VALIDATION] Found {len(missing)} interactors WITHOUT functions:", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        
        # Group by interaction type
        direct_missing = [m for m in missing if m.get("interaction_type") == "direct"]
        indirect_missing = [m for m in missing if m.get("interaction_type") == "indirect"]
        unknown_missing = [m for m in missing if m.get("interaction_type") not in ["direct", "indirect"]]
        
        if direct_missing:
            print(f"\n[VALIDATION] DIRECT interactors missing functions ({len(direct_missing)}):", file=sys.stderr)
            for m in direct_missing:
                note = f" - {m['note']}" if m.get('note') else ""
                print(f"  - {m['name']}{note}", file=sys.stderr)
            print(f"  → Should have been processed by Steps 2a-2a5 or Step 2b3", file=sys.stderr)
        
        if indirect_missing:
            print(f"\n[VALIDATION] INDIRECT interactors missing functions ({len(indirect_missing)}):", file=sys.stderr)
            for m in indirect_missing:
                note = f" - {m['note']}" if m.get('note') else ""
                print(f"  - {m['name']}{note}", file=sys.stderr)
            print(f"  → Should have been processed by Step 2b2 (indirect functions)", file=sys.stderr)
        
        if unknown_missing:
            print(f"\n[VALIDATION] UNKNOWN type interactors missing functions ({len(unknown_missing)}):", file=sys.stderr)
            for m in unknown_missing:
                note = f" - {m['note']}" if m.get('note') else ""
                print(f"  - {m['name']}{note}", file=sys.stderr)
            print(f"  → Missing interaction_type classification!", file=sys.stderr)
        
        print(f"\n[VALIDATION] Total interactors in history: {len(interactor_history)}", file=sys.stderr)
        print(f"[VALIDATION] Interactors with functions: {len(interactor_history) - len(missing)}", file=sys.stderr)
        print(f"[VALIDATION] Missing functions: {len(missing)}", file=sys.stderr)
        print(f"[VALIDATION] Completion rate: {((len(interactor_history) - len(missing)) / len(interactor_history) * 100) if interactor_history else 0:.1f}%", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
    else:
        print(f"\n[VALIDATION] [OK]PHASE 2 COMPLETE: All {len(interactor_history)} interactors have functions", file=sys.stderr)


def validate_classification_preservation(
    before_payload: dict,
    after_payload: dict,
    step_name: str
) -> bool:
    """
    Ensure post-processing doesn't corrupt interaction_type classifications.
    
    Args:
        before_payload: Payload before post-processing step
        after_payload: Payload after post-processing step
        step_name: Name of the post-processing step for logging
        
    Returns:
        True if all classifications preserved, False if corruptions detected
    """
    # Extract classifications from before
    before_snap = before_payload.get('snapshot_json', {})
    before_classifications = {
        i.get('primary'): i.get('interaction_type')
        for i in before_snap.get('interactors', [])
        if i.get('primary')
    }
    
    # Extract classifications from after
    after_snap = after_payload.get('snapshot_json', {})
    after_classifications = {
        i.get('primary'): i.get('interaction_type')
        for i in after_snap.get('interactors', [])
        if i.get('primary')
    }
    
    # Find corruptions
    corrupted = []
    for protein, before_type in before_classifications.items():
        after_type = after_classifications.get(protein)
        
        # Check if classification changed
        if before_type != after_type and before_type is not None:
            corrupted.append({
                'protein': protein,
                'before': before_type,
                'after': after_type
            })
    
    # Report results
    if corrupted:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[WARN] WARNING: {step_name} changed {len(corrupted)} classification(s)!", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        for corruption in corrupted:
            print(f"  - {corruption['protein']}: {corruption['before']} → {corruption['after']}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        return False
    else:
        print(f"[VALIDATION] [OK]{step_name}: All classifications preserved", file=sys.stderr)
        return True


def run_pipeline(
    user_query: str,
    verbose: bool = False,
    stream: bool = True,
    num_interactor_rounds: int = 3,
    num_function_rounds: int = 3,
    max_depth: int = 3,
    cancel_event=None,
) -> Dict[str, Any]:
    """Execute the full pipeline with configurable discovery rounds.

    Args:
        user_query: Protein name to analyze
        verbose: Print detailed debugging info
        stream: Enable streaming previews
        num_interactor_rounds: Number of interactor discovery rounds (min 3, max 10)
        num_function_rounds: Number of function mapping rounds (min 3, max 10)
        max_depth: Maximum chain depth for indirect interactors (1-4, or 5+ for unlimited)
    """
    # Generate pipeline with requested rounds
    if DYNAMIC_CONFIG_AVAILABLE:
        pipeline_steps = generate_pipeline(num_interactor_rounds, num_function_rounds, max_depth)
    else:
        pipeline_steps = DEFAULT_PIPELINE_STEPS

    validated_steps = validate_steps(pipeline_steps)
    current_payload: Optional[Dict[str, Any]] = None

    # Token tracking with cost estimates
    # Gemini 2.5 Pro pricing (as of 2025):
    # Input: $1.25 per 1M tokens
    # Output: $10.00 per 1M tokens
    # Thinking: Same as input ($1.25 per 1M tokens)
    pipeline_token_stats = {
        'total_input_tokens': 0,
        'total_thinking_tokens': 0,
        'total_output_tokens': 0,
        'total_tokens': 0,
        'total_cost': 0.0,
        'steps': []
    }

    # Track overall pipeline time
    import time as time_module
    pipeline_start_time = time_module.time()

    # Initialize step logger (only if enabled via environment)
    step_logger = None
    if STEP_LOGGER_AVAILABLE:
        step_logger = StepLogger(user_query)

    print(f"\n{'='*80}")
    print(f"RUNNING MAXIMIZED PIPELINE FOR: {user_query}")
    print(f"{'='*80}")
    print(f"Total steps: {len(validated_steps)}")
    print(f"Thinking budget: Auto (model decides per step)")
    print(f"Max output per step: 65,536 tokens")
    print(f"{'='*80}\n")

    for step_idx, step in enumerate(validated_steps, start=1):
        # Check for cancellation before each step
        if cancel_event and cancel_event.is_set():
            raise PipelineError("Job cancelled by user")

        step_start_time = time_module.time()

        # Log step start
        if step_logger:
            step_logger.log_step_start(
                step_name=step.name,
                input_data=current_payload,
                step_type="pipeline"
            )

        print(f"\n[Step {step_idx}/{len(validated_steps)}] {step.name}")
        print(f"{'-'*80}")

        # Capture terminal output for logging
        if step_logger:
            step_logger.log_terminal_output(f"[Step {step_idx}/{len(validated_steps)}] {step.name}")
            step_logger.log_terminal_output('-' * 80)

        is_first = (step_idx == 1)

        # Special handling for snapshot step
        if step.name == "step3_snapshot":
            if current_payload and "ctx_json" in current_payload:
                print("   Creating snapshot locally (no model call)...")
                current_payload = create_snapshot_from_ctx(
                    current_payload["ctx_json"],
                    list(step.expected_columns),
                    step.name,
                )
                step_elapsed = time_module.time() - step_start_time
                print(f"   [OK]Snapshot created ({step_elapsed:.1f}s)")
            continue

        # Build prompt
        prompt = build_prompt(step, current_payload, user_query, is_first)

        if verbose:
            print("\nPrompt:\n" + prompt + "\n")

        # Call model and get token stats (with cancellation check)
        raw_output, token_stats = call_gemini_model(step, prompt, cancel_event=cancel_event)

        # Log AI response
        if step_logger:
            step_logger.log_ai_response(raw_output, metadata=token_stats)

        if verbose:
            print("Model output:\n" + raw_output + "\n")

        # Parse output and merge differentially
        previous_interactor_count = 0
        previous_function_count = 0
        if current_payload and "ctx_json" in current_payload:
            previous_interactor_count = len(current_payload["ctx_json"].get("interactors", []))
            previous_function_count = sum(
                len(i.get("functions", []))
                for i in current_payload["ctx_json"].get("interactors", [])
            )

        current_payload = parse_json_output(
            raw_output,
            list(step.expected_columns),
            previous_payload=current_payload,
        )

        # Validate no data loss occurred during merge
        if current_payload and "ctx_json" in current_payload:
            new_interactor_count = len(current_payload["ctx_json"].get("interactors", []))
            new_function_count = sum(
                len(i.get("functions", []))
                for i in current_payload["ctx_json"].get("interactors", [])
            )

            # Ensure counts never decrease (unless it's a QC step that removes invalid data)
            if new_interactor_count < previous_interactor_count:
                print(f"   [WARN]WARNING: Interactor count decreased from {previous_interactor_count} to {new_interactor_count}")
            if new_function_count < previous_function_count:
                print(f"   [WARN]WARNING: Function count decreased from {previous_function_count} to {new_function_count}")

        # Track tokens and calculate costs
        thinking_tokens = _coerce_token_count(token_stats.get('thinking_tokens'))
        output_tokens = _coerce_token_count(token_stats.get('output_tokens'))
        total_tokens = _coerce_token_count(token_stats.get('total_tokens'))

        # Calculate input tokens (total - thinking - output)
        input_tokens = max(0, total_tokens - thinking_tokens - output_tokens)

        # Calculate costs (Gemini 2.5 Pro pricing)
        # Input & Thinking: $1.25 per 1M tokens
        # Output: $10.00 per 1M tokens
        input_cost = (input_tokens / 1_000_000) * 1.25
        thinking_cost = (thinking_tokens / 1_000_000) * 1.25
        output_cost = (output_tokens / 1_000_000) * 10.00
        total_cost = input_cost + thinking_cost + output_cost

        # Calculate step elapsed time
        step_elapsed = time_module.time() - step_start_time

        step_stat = {
            'step': step.name,
            'input_tokens': input_tokens,
            'thinking_tokens': thinking_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost': input_cost,
            'thinking_cost': thinking_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'elapsed_time': step_elapsed
        }
        pipeline_token_stats['steps'].append(step_stat)
        pipeline_token_stats['total_input_tokens'] += input_tokens
        pipeline_token_stats['total_thinking_tokens'] += thinking_tokens
        pipeline_token_stats['total_output_tokens'] += output_tokens
        pipeline_token_stats['total_tokens'] += total_tokens
        pipeline_token_stats['total_cost'] += total_cost

        # Show progress with token info
        interactor_count = 0
        total_functions = 0
        if current_payload and "ctx_json" in current_payload:
            interactor_count = len(current_payload["ctx_json"].get("interactors", []))
            total_functions = sum(
                len(i.get("functions", []))
                for i in current_payload["ctx_json"].get("interactors", [])
            )

            print(f"  → {interactor_count} interactors, {total_functions} functions mapped")

            # Capture terminal output
            if step_logger:
                step_logger.log_terminal_output(f"  → {interactor_count} interactors, {total_functions} functions mapped")

        # Print token usage and cost for this step
        print(f"  → Tokens: input={input_tokens:,}, thinking={thinking_tokens:,}, output={output_tokens:,}, total={total_tokens:,}")
        print(f"  → Cost: ${total_cost:.4f} (input: ${input_cost:.4f}, thinking: ${thinking_cost:.4f}, output: ${output_cost:.4f})")
        print(f"  → Time: {step_elapsed:.1f}s")

        # Capture terminal output
        if step_logger:
            step_logger.log_terminal_output(f"  → Tokens: input={input_tokens:,}, thinking={thinking_tokens:,}, output={output_tokens:,}, total={total_tokens:,}")
            step_logger.log_terminal_output(f"  → Cost: ${total_cost:.4f} (input: ${input_cost:.4f}, thinking: ${thinking_cost:.4f}, output: ${output_cost:.4f})")
            step_logger.log_terminal_output(f"  → Time: {step_elapsed:.1f}s")

        # Log step completion
        if step_logger:
            step_metadata = {
                'step_name': step.name,
                'interactor_count': interactor_count,
                'function_count': total_functions,
                'input_tokens': input_tokens,
                'thinking_tokens': thinking_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'input_cost': input_cost,
                'thinking_cost': thinking_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }
            step_logger.log_step_complete(
                output_data=current_payload,
                metadata=step_metadata,
                generate_summary=True
            )

    if current_payload is None:
        raise PipelineError("Pipeline completed without returning data.")

    # Calculate total pipeline time
    pipeline_elapsed = time_module.time() - pipeline_start_time
    pipeline_elapsed_min = pipeline_elapsed / 60

    # Log final pipeline output (before post-processing)
    if step_logger:
        step_logger.log_final_output(current_payload)

    # Print comprehensive token and cost summary
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {pipeline_elapsed_min:.1f} minutes ({pipeline_elapsed:.0f}s)")
    print(f"\n{'='*80}")
    print("TOKEN USAGE & COST BREAKDOWN")
    print(f"{'='*80}")
    print(f"\nTOTAL TOKENS:")
    print(f"  Input tokens:    {pipeline_token_stats['total_input_tokens']:>12,}")
    print(f"  Thinking tokens: {pipeline_token_stats['total_thinking_tokens']:>12,}")
    print(f"  Output tokens:   {pipeline_token_stats['total_output_tokens']:>12,}")
    print(f"  {'─'*40}")
    print(f"  TOTAL:           {pipeline_token_stats['total_tokens']:>12,}")

    print(f"\nESTIMATED COST (Gemini 2.5 Pro):")
    input_total_cost = (pipeline_token_stats['total_input_tokens'] / 1_000_000) * 1.25
    thinking_total_cost = (pipeline_token_stats['total_thinking_tokens'] / 1_000_000) * 1.25
    output_total_cost = (pipeline_token_stats['total_output_tokens'] / 1_000_000) * 10.00
    print(f"  Input:    ${input_total_cost:>8.4f}")
    print(f"  Thinking: ${thinking_total_cost:>8.4f}")
    print(f"  Output:   ${output_total_cost:>8.4f}")
    print(f"  {'─'*24}")
    print(f"  TOTAL:    ${pipeline_token_stats['total_cost']:>8.4f}")

    print(f"\n{'='*80}")
    print("PER-STEP BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Step':<32} {'Input':>8} {'Think':>8} {'Output':>8} {'Total':>10} {'Cost':>10} {'Time':>8}")
    print(f"{'-'*80}")
    for step_stat in pipeline_token_stats['steps']:
        print(f"{step_stat['step']:<32} "
              f"{step_stat['input_tokens']:>8,} "
              f"{step_stat['thinking_tokens']:>8,} "
              f"{step_stat['output_tokens']:>8,} "
              f"{step_stat['total_tokens']:>10,} "
              f"${step_stat['total_cost']:>9.4f} "
              f"{step_stat['elapsed_time']:>7.1f}s")
    print(f"{'-'*80}")
    print(f"{'TOTAL':<32} "
          f"{pipeline_token_stats['total_input_tokens']:>8,} "
          f"{pipeline_token_stats['total_thinking_tokens']:>8,} "
          f"{pipeline_token_stats['total_output_tokens']:>8,} "
          f"{pipeline_token_stats['total_tokens']:>10,} "
          f"${pipeline_token_stats['total_cost']:>9.4f} "
          f"{pipeline_elapsed:>7.0f}s")
    print(f"{'='*80}\n")

    # Return both payload and step_logger for post-processing
    if STEP_LOGGER_AVAILABLE and step_logger:
        return current_payload, step_logger
    return current_payload, None


# ============================================================================
# FLASK-COMPATIBLE WEB INTEGRATION (NEW)
# ============================================================================

def _get_user_friendly_step_name(step_name: str) -> str:
    """
    Convert internal pipeline step names to user-friendly display text.

    Args:
        step_name: Internal step identifier (e.g., "step1a_discover")

    Returns:
        User-friendly description for display
    """
    # Interactor discovery steps (step1*)
    if step_name == "step1a_discover":
        return "Researching interactors..."
    elif step_name == "step1b_expand":
        return "Expanding interaction network..."
    elif step_name == "step1c_deep_mining":
        return "Deep mining literature for interactors..."
    elif "step1d" in step_name or "round2" in step_name.lower():
        return "Round 2: Discovering additional interactors..."
    elif "step1e" in step_name or "round3" in step_name.lower():
        return "Round 3: Finding more interactors..."
    elif "step1f" in step_name or "round4" in step_name.lower():
        return "Round 4: Expanding interactor search..."
    elif "step1g" in step_name or "round5" in step_name.lower():
        return "Round 5: Comprehensive interactor sweep..."
    elif step_name.startswith("step1"):
        # Catch-all for any other step1* variants
        return "Discovering protein interactors..."

    # Function mapping steps (step2*)
    elif step_name == "step2a_functions":
        return "Mapping biological functions..."
    elif step_name == "step2a2_functions_batch":
        return "Analyzing additional functions..."
    elif step_name == "step2a3_functions_exhaustive":
        return "Comprehensive function analysis..."
    elif "step2a4" in step_name or ("step2" in step_name and "round2" in step_name.lower()):
        return "Round 2: Discovering additional functions..."
    elif "step2a5" in step_name or ("step2" in step_name and "round3" in step_name.lower()):
        return "Round 3: Finding more functions..."
    elif step_name == "step2b_deep_function_research":
        return "Deep research on biological functions..."
    elif step_name.startswith("step2c_arrow_"):
        # NEW: Arrow determination steps (dynamically generated)
        # Extract interactor name from step_name (e.g., "step2c_arrow_VCP" -> "VCP")
        interactor = step_name.replace("step2c_arrow_", "")
        return f"Determining arrow/direction for {interactor}..."
    elif step_name == "step2g_final_qc":
        return "Final quality control..."
    elif step_name.startswith("step2"):
        # Catch-all for any other step2* variants
        return "Analyzing biological functions..."

    # Snapshot step (step3)
    elif step_name == "step3_snapshot":
        return "Building network snapshot..."

    # Fallback for unknown steps
    else:
        # Convert underscores to spaces and capitalize for basic readability
        return step_name.replace("_", " ").title()


def _run_main_pipeline_for_web(
    user_query: str,
    update_status_func,
    total_steps: int,
    num_interactor_rounds: int = 3,
    num_function_rounds: int = 3,
    max_depth: int = 3,
    cancel_event=None,
    known_interactions: Optional[List[Dict[str, Any]]] = None,
    skip_arrow_determination: bool = False,
) -> Dict[str, Any]:
    """
    A lean, web-focused version of run_pipeline. It only runs the main data
    gathering steps and reports progress. It does NOT handle validation or file saving.

    NEW: Dynamically generates arrow determination steps (Step 2c) after function discovery.

    Args:
        user_query: Protein name
        update_status_func: Function to update progress
        total_steps: Total number of steps including post-processing (for accurate progress)
        num_interactor_rounds: Number of interactor rounds
        num_function_rounds: Number of function rounds
        max_depth: Maximum chain depth for indirect interactors
        cancel_event: Optional threading.Event to check for cancellation
        known_interactions: List of known interactions from database (for exclusion)
    """
    # Generate pipeline with requested rounds
    if DYNAMIC_CONFIG_AVAILABLE:
        pipeline_steps = generate_pipeline(num_interactor_rounds, num_function_rounds, max_depth)
    else:
        pipeline_steps = DEFAULT_PIPELINE_STEPS

    validated_steps = validate_steps(pipeline_steps)
    current_payload: Optional[Dict[str, Any]] = None
    arrow_steps_executed = False  # Track if we've already done arrow determination

    for step_idx, step in enumerate(validated_steps, start=1):
        # Check for cancellation before each step
        if cancel_event and cancel_event.is_set():
            raise PipelineError("Job cancelled by user")

        # Report progress to the web UI with user-friendly name
        friendly_name = _get_user_friendly_step_name(step.name)
        update_status_func(
            text=friendly_name,
            current_step=step_idx,
            total_steps=total_steps
        )

        if step.name == "step3_snapshot":
            if current_payload and "ctx_json" in current_payload:
                current_payload = create_snapshot_from_ctx(
                    current_payload["ctx_json"],
                    list(step.expected_columns),
                    step.name,
                )
            continue

        # Build prompt with known interactions context for exclusion
        prompt = build_prompt(
            step,
            current_payload,
            user_query,
            (step_idx == 1),
            known_interactions=known_interactions
        )
        # Note: We discard the token_stats as they aren't used in the web UI
        raw_output, _ = call_gemini_model(step, prompt, cancel_event=cancel_event)

        current_payload = parse_json_output(
            raw_output,
            list(step.expected_columns),
            previous_payload=current_payload,
        )

        # ===================================================================
        # ARROW DETERMINATION (Heuristic Fallback)
        # ===================================================================
        # We rely on the function mapping steps (2a, 2b) to provide arrows.
        # If they are missing, we apply a fast heuristic fallback.
        if (step.name == "step2b2_indirect_functions" and
            not arrow_steps_executed and
            current_payload and "ctx_json" in current_payload):

            arrow_steps_executed = True
            
            # Check if we need to apply heuristics
            ctx_json = current_payload.get("ctx_json", {})
            interactors = ctx_json.get("interactors", [])
            needs_heuristic = False
            
            for i in interactors:
                if not i.get("arrow") or i.get("arrow") == "binds":
                    # Check if functions imply a direction
                    functions = i.get("functions", [])
                    if functions:
                         needs_heuristic = True
                         break
            
            if needs_heuristic:
                print(f"\n[ARROW DETERMINATION] Applying fast heuristic...", file=sys.stderr)

                # Import heuristic function
                from utils.interaction_metadata_generator import determine_interaction_arrow
                from collections import Counter

                for interactor in interactors:
                    functions = interactor.get("functions", [])
                    if not functions:
                        continue

                    # Use heuristic to determine arrow if not already set or generic
                    if not interactor.get("arrow") or interactor.get("arrow") == "binds":
                        arrow = determine_interaction_arrow(functions)
                        interactor["arrow"] = arrow

                    # Majority vote for direction
                    directions = [f.get("direction", "bidirectional") for f in functions]
                    if directions:
                        most_common = Counter(directions).most_common(1)[0][0]
                        interactor["direction"] = most_common
                    else:
                        if not interactor.get("direction"):
                             interactor["direction"] = "bidirectional"

                    # Set default intent if missing
                    if not interactor.get("intent"):
                        interactor["intent"] = "regulation"

                print(f"[ARROW DETERMINATION] Heuristic applied to interactors\n", file=sys.stderr)

    if current_payload is None:
        raise PipelineError("Main pipeline completed without returning data.")

    return current_payload


def run_full_job(
    user_query: str,
    jobs: dict,
    lock: Lock,
    num_interactor_rounds: int = 3,
    num_function_rounds: int = 3,
    max_depth: int = 3,
    skip_validation: bool = False,
    skip_deduplicator: bool = False,
    skip_arrow_determination: bool = False,
    skip_fact_checking: bool = False,
    flask_app = None
):
    """
    This is the master function for the Flask background thread.
    It orchestrates the main pipeline AND the evidence validation.

    Args:
        user_query: Protein name to analyze
        jobs: Shared jobs dictionary for status tracking
        lock: Threading lock for jobs dict access
        num_interactor_rounds: Number of interactor discovery rounds (default: 3)
        num_function_rounds: Number of function mapping rounds (default: 3)
        max_depth: Maximum chain depth for indirect interactors (1-4, or 5+ for unlimited)
        skip_validation: Skip evidence validation step
        skip_deduplicator: Skip function deduplication step
        skip_arrow_determination: Skip LLM arrow determination, use heuristic (100× faster)
        skip_fact_checking: Skip claim fact-checking step (faster, may include unverified claims)
        flask_app: Flask app instance (required for database operations in background thread)
    """
    # Get the cancel_event from jobs dict
    cancel_event = None
    with lock:
        if user_query in jobs:
            cancel_event = jobs[user_query].get('cancel_event')

    def update_status(text: str, current_step: int = None, total_steps: int = None):
        with lock:
            # Only update if this is still our job (check cancel_event identity)
            if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                progress_update = {"text": text}
                if current_step and total_steps:
                    progress_update.update({"current": current_step, "total": total_steps})
                jobs[user_query]['progress'] = progress_update

    # ========================================================================
    # CALCULATE TOTAL STEPS (before starting work)
    # ========================================================================
    # Generate pipeline to get accurate step count
    if DYNAMIC_CONFIG_AVAILABLE:
        pipeline_steps = generate_pipeline(num_interactor_rounds, num_function_rounds, max_depth)
    else:
        pipeline_steps = DEFAULT_PIPELINE_STEPS

    # Count pipeline steps
    pipeline_step_count = len(pipeline_steps)

    # Count post-processing steps (conditional based on availability and flags)
    api_key = os.getenv("GOOGLE_API_KEY") or ""
    post_steps = 0

    # Evidence validation (conditional)
    if not skip_validation and VALIDATOR_AVAILABLE:
        post_steps += 1

    # Metadata generation (conditional)
    if METADATA_GENERATOR_AVAILABLE and generate_interaction_metadata is not None:
        post_steps += 1

    # Schema validation - pre-gate (conditional)
    if SCHEMA_VALIDATOR_AVAILABLE and validate_schema_consistency is not None:
        post_steps += 1

    # Deduplication (conditional)
    if not skip_deduplicator and DEDUPLICATOR_AVAILABLE and deduplicate_payload is not None and api_key:
        post_steps += 1

    # Fact-checking (conditional)
    if not skip_fact_checking and FACT_CHECKER_AVAILABLE and api_key:
        post_steps += 1

    # Schema validation - post-finalization (conditional)
    if SCHEMA_VALIDATOR_AVAILABLE and finalize_interaction_metadata is not None:
        post_steps += 1

    # PMID updates (conditional)
    if PMID_UPDATER_AVAILABLE and update_payload_pmids is not None:
        post_steps += 1

    # Arrow effect validation + direct link extraction (conditional)
    if ARROW_VALIDATOR_AVAILABLE and validate_arrows_for_payload is not None and api_key:
        post_steps += 1

    # Function name cleaning (conditional)
    if FUNCTION_NAME_CLEANER_AVAILABLE and clean_payload_function_names is not None:
        post_steps += 1

    # Final stages (ALWAYS run)
    post_steps += 3  # Database save + File cache + PostgreSQL sync

    # Total steps = pipeline + post-processing
    total_steps = pipeline_step_count + post_steps
    current_step = 0

    print(f"[PROGRESS] Total steps calculated: {total_steps} (pipeline: {pipeline_step_count}, post: {post_steps})", file=sys.stderr)

    # Initialize step logger for post-processing (only if enabled via environment)
    step_logger = None
    if STEP_LOGGER_AVAILABLE:
        step_logger = StepLogger(user_query)

    try:
        # --- STAGE 0: Load known interactions from PostgreSQL database ---
        update_status("Loading known interactions from database...")
        known_interactions = []

        try:
            if flask_app is not None:
                with flask_app.app_context():
                    from models import Protein, Interaction
                    from app import db

                    protein_obj = Protein.query.filter_by(symbol=user_query).first()

                    if protein_obj:
                        # Get all interactions (bidirectional due to canonical ordering)
                        db_interactions = db.session.query(Interaction).filter(
                            (Interaction.protein_a_id == protein_obj.id) |
                            (Interaction.protein_b_id == protein_obj.id)
                        ).all()

                        # Convert to interactor format for exclusion context
                        for interaction in db_interactions:
                            if interaction.protein_a_id == protein_obj.id:
                                partner = interaction.protein_b
                            else:
                                partner = interaction.protein_a

                            known_interactions.append({
                                "primary": partner.symbol,
                                "confidence": interaction.confidence or 0.5,
                                "arrow": interaction.arrow or "binds"
                            })

                        print(f"[PostgreSQL] Found {len(known_interactions)} known interactions for {user_query}", file=sys.stderr)
                    else:
                        print(f"[PostgreSQL] No known interactions found for {user_query} - first query", file=sys.stderr)
            else:
                print(f"[WARNING] Flask app context not available, using file DB fallback", file=sys.stderr)
                known_interactions = pdb.get_all_interactions(user_query)
        except Exception as e:
            print(f"[WARNING] PostgreSQL history query failed: {e}, using file DB fallback", file=sys.stderr)
            known_interactions = pdb.get_all_interactions(user_query)

        if known_interactions:
            print(f"[DB] History loaded: {len(known_interactions)} known interactions", file=sys.stderr)

        # --- STAGE 1: Run the main pipeline with known interactions context ---
        pipeline_payload = _run_main_pipeline_for_web(
            user_query,
            update_status,
            total_steps=total_steps,  # Pass accurate total
            num_interactor_rounds=num_interactor_rounds,
            num_function_rounds=num_function_rounds,
            max_depth=max_depth,
            cancel_event=cancel_event,
            known_interactions=known_interactions,  # Pass to pipeline for exclusion
            skip_arrow_determination=skip_arrow_determination,
        )

        # Pipeline completed - update current step
        current_step = pipeline_step_count

        # --- STAGE 2: Schema Consistency Pre-Gate (normalize + dedupe baseline) ---
        if SCHEMA_VALIDATOR_AVAILABLE and validate_schema_consistency is not None:
            current_step += 1
            update_status(
                text="Validating data consistency...",
                current_step=current_step,
                total_steps=total_steps
            )
            schema_ok_payload = validate_schema_consistency(
                pipeline_payload,
                fix_arrows=True,
                fix_chains=True,
                fix_directions=True,
                verbose=False
            )
        else:
            schema_ok_payload = pipeline_payload

        # --- STAGE 3: Deduplicate functions (AI) BEFORE evidence validation ---
        if skip_deduplicator:
            deduped_payload = schema_ok_payload
        elif DEDUPLICATOR_AVAILABLE and deduplicate_payload is not None and api_key:
            current_step += 1
            update_status(
                text="Deduplicating functions...",
                current_step=current_step,
                total_steps=total_steps
            )
            deduped_payload = deduplicate_payload(
                schema_ok_payload,
                api_key,
                verbose=False
            )
        else:
            deduped_payload = schema_ok_payload

        # --- STAGE 4: Evidence validation/enrichment (after structure cleanup) ---
        if skip_validation:
            validated_payload = deduped_payload
        elif not VALIDATOR_AVAILABLE:
            validated_payload = deduped_payload
        else:
            current_step += 1
            update_status(
                text="Validating & enriching evidence...",
                current_step=current_step,
                total_steps=total_steps
            )
            validated_payload = validate_and_enrich_evidence(
                deduped_payload,
                api_key,
                verbose=False,
                step_logger=step_logger
            )

        # --- STAGE 5: Interaction-level metadata synthesis (uses enriched evidence) ---
        if METADATA_GENERATOR_AVAILABLE and generate_interaction_metadata is not None:
            current_step += 1
            update_status(
                text="Analyzing interaction patterns...",
                current_step=current_step,
                total_steps=total_steps
            )
            validated_payload = generate_interaction_metadata(
                validated_payload,
                verbose=False
            )

        # --- STAGE 6: Claim fact-checker ---
        if not skip_fact_checking and FACT_CHECKER_AVAILABLE and api_key:
            current_step += 1
            update_status(
                text="Fact-checking claims with Google Search...",
                current_step=current_step,
                total_steps=total_steps
            )
            fact_checked_payload = fact_check_json(
                validated_payload,
                api_key,
                verbose=False
            )
        else:
            fact_checked_payload = validated_payload

        # --- STAGE 6B: Second deduplication pass (catch fact-checker-created dupes) ---
        if not skip_fact_checking and DEDUPLICATOR_AVAILABLE and api_key and FACT_CHECKER_AVAILABLE:
            current_step += 1
            update_status(
                text="Running final deduplication pass...",
                current_step=current_step,
                total_steps=total_steps
            )
            final_deduplicated_payload = deduplicate_payload(
                fact_checked_payload,
                api_key,
                verbose=False
            )
        else:
            final_deduplicated_payload = fact_checked_payload

        # --- STAGE 7: Update PMIDs (finalize citations and prune empty entries) ---
        if PMID_UPDATER_AVAILABLE and update_payload_pmids is not None:
            current_step += 1
            update_status(
                text="Validating citations...",
                current_step=current_step,
                total_steps=total_steps
            )
            final_payload = update_payload_pmids(
                final_deduplicated_payload,
                verbose=False
            )
        else:
            final_payload = final_deduplicated_payload

# --- STAGE 7.5: Validate arrows, directions, effects + extract direct mediator links ---
        if ARROW_VALIDATOR_AVAILABLE and validate_arrows_for_payload is not None and api_key:
            current_step += 1
            update_status(
                text="Validating arrows & extracting direct links...",
                current_step=current_step,
                total_steps=total_steps
            )
            final_payload = validate_arrows_for_payload(
                final_payload,
                api_key,
                verbose=False  # <--- FIXED: Hardcoded to False
            )
            
        # --- STAGE 8: Clean function names (remove generic terms) ---
        if FUNCTION_NAME_CLEANER_AVAILABLE and clean_payload_function_names is not None:
            current_step += 1
            update_status(
                text="Normalizing function names...",
                current_step=current_step,
                total_steps=total_steps
            )
            final_payload = clean_payload_function_names(
                final_payload,
                verbose=False
            )

        # --- STAGE 9: Finalize interaction metadata and sync snapshot ---
        if SCHEMA_VALIDATOR_AVAILABLE and finalize_interaction_metadata is not None:
            current_step += 1
            update_status(
                text="Finalizing interaction metadata...",
                current_step=current_step,
                total_steps=total_steps
            )
            final_payload = finalize_interaction_metadata(
                final_payload,
                add_arrow_notation=True,
                validate_snapshot=True,
                verbose=False
            )

        # --- STAGE 10: Save to protein-interaction database ---
        current_step += 1
        update_status(
            text="Building knowledge graph...",
            current_step=current_step,
            total_steps=total_steps
        )
        snapshot = final_payload.get("snapshot_json", {})
        new_interactors = snapshot.get("interactors", [])

        # Save each interaction to database (symmetric storage)
        saved_count = 0
        for interactor in new_interactors:
            partner = interactor.get("primary")
            if partner:
                success = pdb.save_interaction(user_query, partner, interactor)
                if success:
                    saved_count += 1
                    print(f"[DB] Saved interaction: {user_query} <-> {partner}", file=sys.stderr)

        # Update protein metadata
        pdb.update_protein_metadata(user_query, query_completed=True)
        print(f"[DB] Saved {saved_count} interactions to database", file=sys.stderr)

        # --- STAGE 10.5: Save to OLD cache format for backward compatibility ---
        current_step += 1
        update_status(
            text="Caching results...",
            current_step=current_step,
            total_steps=total_steps
        )
        # File 1: PROTEIN.json - snapshot_json only (for visualization)
        output_path = os.path.join(CACHE_DIR, f"{user_query}.json")
        snapshot_only = {
            "snapshot_json": final_payload.get("snapshot_json", {})
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_only, f, ensure_ascii=False, indent=2)

        # File 2: PROTEIN_metadata.json - ctx_json (full rich metadata)
        metadata_path = os.path.join(CACHE_DIR, f"{user_query}_metadata.json")
        metadata_only = {
            "ctx_json": final_payload.get("ctx_json", {})
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_only, f, ensure_ascii=False, indent=2)

        # --- STAGE 10.75: Sync to NEW PostgreSQL database ---
        current_step += 1
        update_status(
            text="Syncing to database...",
            current_step=current_step,
            total_steps=total_steps
        )

        # Enhanced logging and retry logic for database sync
        from datetime import datetime
        import time

        sync_start_time = datetime.now()
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[DB SYNC] Starting PostgreSQL sync for '{user_query}' at {sync_start_time.strftime('%H:%M:%S')}", file=sys.stderr)
        print(f"[DB SYNC] Flask app provided: {flask_app is not None}", file=sys.stderr)

        if flask_app is None:
            print(f"[ERROR][DB SYNC] Flask app instance is None - CANNOT sync to PostgreSQL!", file=sys.stderr)
            print(f"   Data preserved in file cache only", file=sys.stderr)
            print(f"   Run 'python sync_cache_to_db.py {user_query}' to sync manually", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        else:
            max_retries = 3
            retry_count = 0
            sync_success = False

            while retry_count < max_retries and not sync_success:
                try:
                    from utils.db_sync import DatabaseSyncLayer

                    if retry_count > 0:
                        wait_time = retry_count * 5
                        print(f"[DB SYNC] Retry {retry_count}/{max_retries} in {wait_time}s...", file=sys.stderr)
                        time.sleep(wait_time)

                    # Test connection first
                    with flask_app.app_context():
                        from models import db
                        db.session.execute(db.text('SELECT 1'))
                        print(f"[DB SYNC] [OK]Database connection verified", file=sys.stderr)

                    # CRITICAL: Flask-SQLAlchemy requires app context in background threads
                    with flask_app.app_context():
                        sync_layer = DatabaseSyncLayer()

                        num_interactions = len(final_payload.get("snapshot_json", {}).get("interactors", []))
                        print(f"[DB SYNC] Syncing protein '{user_query}' with {num_interactions} interactions...", file=sys.stderr)

                        db_stats = sync_layer.sync_query_results(
                            protein_symbol=user_query,
                            snapshot_json={"snapshot_json": final_payload.get("snapshot_json", {})},
                            ctx_json=final_payload.get("ctx_json")
                        )

                        sync_duration = (datetime.now() - sync_start_time).total_seconds()
                        print(f"[DB SYNC] [OK]SUCCESS in {sync_duration:.1f}s", file=sys.stderr)
                        print(f"[DB SYNC]   • Protein: {user_query}", file=sys.stderr)
                        print(f"[DB SYNC]   • Interactions created: {db_stats['interactions_created']}", file=sys.stderr)
                        print(f"[DB SYNC]   • Interactions updated: {db_stats['interactions_updated']}", file=sys.stderr)
                        print(f"{'='*60}\n", file=sys.stderr)
                        sync_success = True

                except Exception as db_error:
                    retry_count += 1
                    if retry_count >= max_retries:
                        # Final failure - log details and continue with file cache
                        print(f"\n[ERROR][DB SYNC] FAILED after {max_retries} attempts", file=sys.stderr)
                        print(f"   Error: {db_error}", file=sys.stderr)
                        print(f"   Data preserved in file cache at: cache/{user_query}.json", file=sys.stderr)
                        print(f"   Run 'python sync_cache_to_db.py {user_query}' to sync manually", file=sys.stderr)
                        print(f"{'='*60}\n", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    else:
                        print(f"[WARN] [DB SYNC] Attempt {retry_count} failed: {db_error}", file=sys.stderr)

        # --- STAGE 7: Mark job as complete ---
        with lock:
            # Only update if this is still our job (check cancel_event identity)
            if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                jobs[user_query]['status'] = 'complete'
                jobs[user_query]['progress'] = 'Done'

    except Exception as e:
        error_message = f"Error: {str(e)}"

        # Check if this was a cancellation
        is_cancelled = "cancelled by user" in error_message.lower()

        if is_cancelled:
            print(f"PIPELINE CANCELLED for '{user_query}'", file=sys.stderr)
            with lock:
                # Only update if this is still our job (check cancel_event identity)
                if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                    jobs[user_query]['status'] = 'cancelled'
                    jobs[user_query]['progress'] = {"text": "Cancelled by user"}
        else:
            print(f"PIPELINE ERROR for '{user_query}': {error_message}", file=sys.stderr)
            # Also print the full traceback for detailed debugging
            import traceback
            traceback.print_exc(file=sys.stderr)

            with lock:
                # Only update if this is still our job (check cancel_event identity)
                if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                    jobs[user_query]['status'] = 'error'
                    jobs[user_query]['progress'] = {"text": error_message}
                    print(f"Successfully updated jobs dictionary for '{user_query}' to error state.", file=sys.stderr)


def run_requery_job(
    user_query: str,
    jobs: dict,
    lock: Lock,
    num_interactor_rounds: int = 3,
    num_function_rounds: int = 3,
    max_depth: int = 3,
    skip_deduplicator: bool = False,
    skip_fact_checking: bool = False,
    flask_app = None
):
    """
    Re-query pipeline that finds ONLY NEW interactors and adds them to existing data.

    This function:
    1. Loads existing cached results
    2. Runs FRESH pipeline with context of what to avoid
    3. Validates ONLY new data
    4. Fact-checks ONLY new data (if not skipped)
    5. Merges new validated data with existing
    6. Saves merged results

    Args:
        user_query: Protein name to re-query
        jobs: Shared jobs dictionary for status tracking
        lock: Threading lock for jobs dict access
        num_interactor_rounds: Number of interactor discovery rounds (default: 3, min: 1)
        num_function_rounds: Number of function mapping rounds (default: 3, min: 1)
        max_depth: Maximum chain depth for indirect interactors (1-4, or 5+ for unlimited)
        skip_deduplicator: Skip function deduplication step
        flask_app: Flask app instance (required for database operations in background thread)
    """
    # Get the cancel_event from jobs dict
    cancel_event = None
    with lock:
        if user_query in jobs:
            cancel_event = jobs[user_query].get('cancel_event')

    def update_status(text: str, current_step: int = None, total_steps: int = None):
        with lock:
            # Only update if this is still our job (check cancel_event identity)
            if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                progress_update = {"text": text}
                if current_step and total_steps:
                    progress_update.update({"current": current_step, "total": total_steps})
                jobs[user_query]['progress'] = progress_update

    # ========================================================================
    # CALCULATE TOTAL STEPS (before starting work)
    # ========================================================================
    # Generate pipeline to get accurate step count
    if DYNAMIC_CONFIG_AVAILABLE:
        # Allow 1-8 rounds for re-queries
        num_interactor_rounds = max(1, min(8, num_interactor_rounds))
        num_function_rounds = max(1, min(8, num_function_rounds))
        pipeline_steps = generate_pipeline(num_interactor_rounds, num_function_rounds, max_depth)
    else:
        pipeline_steps = DEFAULT_PIPELINE_STEPS

    validated_steps = validate_steps(pipeline_steps)
    pipeline_step_count = len(validated_steps)

    # Count post-processing steps (conditional based on availability and flags)
    api_key = os.getenv("GOOGLE_API_KEY") or ""
    post_steps = 0

    # Validation (conditional)
    if VALIDATOR_AVAILABLE:
        post_steps += 1

    # Metadata generation (conditional)
    if METADATA_GENERATOR_AVAILABLE and generate_interaction_metadata is not None:
        post_steps += 1

    # PMID updates (conditional)
    if PMID_UPDATER_AVAILABLE and update_payload_pmids is not None:
        post_steps += 1

    # Deduplication (conditional)
    if not skip_deduplicator and DEDUPLICATOR_AVAILABLE and deduplicate_payload is not None:
        post_steps += 1

    # Fact-checking (conditional)
    if not skip_fact_checking and FACT_CHECKER_AVAILABLE:
        post_steps += 1

    # Final stages (ALWAYS run)
    post_steps += 2  # Merge + Save

    # Total steps = pipeline + post-processing
    total_steps = pipeline_step_count + post_steps
    current_step = 0

    print(f"[RE-QUERY PROGRESS] Total steps calculated: {total_steps} (pipeline: {pipeline_step_count}, post: {post_steps})", file=sys.stderr)

    try:
        # --- STAGE 0: Load existing cache (with backward compatibility) ---
        cache_path = os.path.join(CACHE_DIR, f"{user_query}.json")
        metadata_path = os.path.join(CACHE_DIR, f"{user_query}_metadata.json")

        if not os.path.exists(cache_path):
            raise PipelineError(f"No existing cache found for {user_query}. Run initial query first.")

        update_status("Loading existing results...")

        # Try to load from split files (new format) or single file (old format)
        if os.path.exists(metadata_path):
            # NEW FORMAT: Load from both files (snapshot + metadata)
            print(f"Re-query: Loading from new split-file format", file=sys.stderr)

            with open(cache_path, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)

            # Combine both for compatibility with existing logic
            existing_payload = {
                "snapshot_json": snapshot_data.get("snapshot_json", {}),
                "ctx_json": metadata_data.get("ctx_json", {})
            }
        else:
            # OLD FORMAT: Load from single file (backward compatibility)
            print(f"Re-query: Loading from old single-file format", file=sys.stderr)

            with open(cache_path, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)

            # Extract both parts from the combined file
            existing_payload = {
                "snapshot_json": combined_data.get("snapshot_json", {}),
                "ctx_json": combined_data.get("ctx_json", {})
            }

            # If ctx_json is missing, try to extract it from the root level (very old format)
            if not existing_payload["ctx_json"] and "interactors" in combined_data:
                print(f"Re-query: WARNING - Very old format detected, attempting migration", file=sys.stderr)
                existing_payload["ctx_json"] = {
                    "main": combined_data.get("main", user_query),
                    "interactors": combined_data.get("interactors", []),
                    "interactor_history": [],
                    "function_history": {},
                    "function_batches": []
                }
                if not existing_payload["snapshot_json"]:
                    existing_payload["snapshot_json"] = {
                        "main": combined_data.get("main", user_query),
                        "interactors": combined_data.get("interactors", [])
                    }

        # Extract existing interactors and functions to provide context
        existing_ctx = existing_payload.get("ctx_json", {})
        existing_interactors = existing_ctx.get("interactors", [])
        existing_symbols = [i.get("primary", "") for i in existing_interactors if i.get("primary")]
        existing_function_history = existing_ctx.get("function_history", {})

        print(f"Re-query: Found {len(existing_symbols)} existing interactors", file=sys.stderr)
        print(f"Re-query: Function history for {len(existing_function_history)} proteins", file=sys.stderr)

        # --- STAGE 1: Run FRESH pipeline with context ---
        # (validated_steps and total_steps already calculated above)

        # Initialize payload with existing context (interactor_history and function_history)
        # This allows the AI to see what has already been found
        current_payload: Optional[Dict[str, Any]] = {
            "ctx_json": {
                "main": user_query,
                "interactor_history": existing_ctx.get("interactor_history", []),
                "function_history": existing_function_history,
                "function_batches": existing_ctx.get("function_batches", [])
            }
        }

        # Build context instruction for interactor discovery
        interactor_context_text = f"\n\n**RE-QUERY CONTEXT:**\n"
        interactor_context_text += f"You have previously found these interactors for {user_query}:\n"
        interactor_context_text += f"{', '.join(existing_symbols)}\n\n"
        interactor_context_text += f"**PRIORITY: Find COMPLETELY NEW interactors that are NOT in the above list.**\n"
        interactor_context_text += f"**If you can't find new interactors, you may research the existing ones for NEW FUNCTIONS.**\n"
        interactor_context_text += f"But focus primarily on discovering new interactor proteins first.\n\n"

        # Build detailed context for function discovery with triplet-based avoidance
        function_context_text = f"\n\n**RE-QUERY FUNCTION CONTEXT - TRIPLET-BASED DUPLICATE AVOIDANCE:**\n\n"

        function_context_text += f"**CRITICAL: UNDERSTAND THE TRIPLET MODEL**\n"
        function_context_text += f"A function is ONLY a duplicate if ALL THREE elements match:\n"
        function_context_text += f"  1. Main protein: {user_query}\n"
        function_context_text += f"  2. The SPECIFIC interactor protein\n"
        function_context_text += f"  3. The SPECIFIC function name\n\n"

        function_context_text += f"This means:\n"
        function_context_text += f"[OK]ALLOWED: Same interactor + DIFFERENT function (e.g., VCP already has 'DNA Repair', but 'Cell Cycle' is NEW)\n"
        function_context_text += f"[OK]ALLOWED: Different interactor + SAME function (e.g., VCP has 'DNA Repair', but UBQLN2 + 'DNA Repair' is NEW)\n"
        function_context_text += f"✗ BLOCKED: Same interactor + SAME function (e.g., VCP already has 'DNA Repair', so VCP + 'DNA Repair' again is duplicate)\n\n"

        function_context_text += f"**EXISTING FUNCTION TRIPLETS TO AVOID:**\n"
        function_context_text += f"Below is what each interactor ALREADY does with {user_query}. Only avoid the EXACT combinations listed.\n\n"

        for protein, funcs in existing_function_history.items():
            if funcs:
                function_context_text += f"━━━ {user_query} + {protein} ━━━\n"
                function_context_text += f"This specific interaction already covers {len(funcs)} function(s):\n"
                for func_name in funcs:
                    function_context_text += f"  ✗ AVOID: ({user_query}, {protein}, \"{func_name}\")\n"
                function_context_text += f"  [OK]BUT: ({user_query}, {protein}, <any NEW function>) is ALLOWED\n"
                function_context_text += "\n"

        function_context_text += f"**YOUR MISSION:**\n"
        function_context_text += f"1. For EXISTING interactors above: Find NEW functions they perform with {user_query}\n"
        function_context_text += f"2. For NEW interactors (not listed above): Find ALL their functions with {user_query}\n"
        function_context_text += f"3. ONLY avoid the exact triplets marked with ✗ above\n"
        function_context_text += f"4. If you find a function name in the list, check WHICH INTERACTOR it's paired with - if it's a different interactor, it's NEW!\n\n"

        function_context_text += f"**EXAMPLES - WHAT TO ADD:**\n"
        function_context_text += f"If ({user_query}, VCP, 'DNA Repair') exists:\n"
        function_context_text += f"  [OK]ADD: ({user_query}, VCP, 'Telomere Maintenance') - different function, same interactor\n"
        function_context_text += f"  [OK]ADD: ({user_query}, VCP, 'Cell Cycle Regulation') - different function, same interactor\n"
        function_context_text += f"  [OK]ADD: ({user_query}, UBQLN2, 'DNA Repair') - same function, different interactor\n"
        function_context_text += f"  ✗ SKIP: ({user_query}, VCP, 'DNA Repair') - exact duplicate\n"
        function_context_text += f"  ✗ SKIP: ({user_query}, VCP, 'DNA Damage Repair') - semantic duplicate of 'DNA Repair'\n\n"

        for step_idx, step in enumerate(validated_steps, start=1):
            # Check for cancellation before each step
            if cancel_event and cancel_event.is_set():
                raise PipelineError("Job cancelled by user")

            # Report progress with user-friendly name
            friendly_name = _get_user_friendly_step_name(step.name)
            update_status(
                text=f"Re-query: {friendly_name}",
                current_step=step_idx,
                total_steps=total_steps
            )

            if step.name == "step3_snapshot":
                if current_payload and "ctx_json" in current_payload:
                    current_payload = create_snapshot_from_ctx(
                        current_payload["ctx_json"],
                        list(step.expected_columns),
                        step.name,
                    )
                continue

            # Build prompt
            prompt = build_prompt(step, current_payload, user_query, (step_idx == 1))

            # Add appropriate context based on step type
            if "discover" in step.name.lower() or "step1" in step.name:
                # Interactor discovery step - add interactor context
                prompt += interactor_context_text
            elif "function" in step.name.lower() or "step2a" in step.name:
                # Function mapping step - add function context
                prompt += function_context_text

            raw_output, _ = call_gemini_model(step, prompt, cancel_event=cancel_event)

            current_payload = parse_json_output(
                raw_output,
                list(step.expected_columns),
                previous_payload=current_payload,
            )

        if current_payload is None:
            raise PipelineError("Re-query pipeline completed without returning data.")

        new_pipeline_payload = current_payload

        # Pipeline completed - update current step
        current_step = pipeline_step_count

        # Extract NEW interactors and updates to existing ones
        new_ctx = new_pipeline_payload.get("ctx_json", {})
        new_interactors = new_ctx.get("interactors", [])
        new_symbols = [i.get("primary", "") for i in new_interactors if i.get("primary")]

        # Separate truly new interactors from updates to existing ones
        truly_new_interactors = []
        updated_existing_interactors = []

        for interactor in new_interactors:
            primary = interactor.get("primary")
            if not primary:
                continue

            if primary in existing_symbols:
                # This is an update to an existing interactor (likely new functions)
                updated_existing_interactors.append(interactor)
            else:
                # This is a completely new interactor
                truly_new_interactors.append(interactor)

        print(f"Re-query: Pipeline found {len(new_interactors)} interactors", file=sys.stderr)
        print(f"Re-query: {len(truly_new_interactors)} are truly new, {len(updated_existing_interactors)} are updates to existing", file=sys.stderr)

        # Combine for validation (both new and updates need validation)
        interactors_to_validate = truly_new_interactors + updated_existing_interactors

        if not interactors_to_validate:
            # No new data found
            update_status("No new data found. Search complete.")
            with lock:
                if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                    jobs[user_query]['status'] = 'complete'
                    jobs[user_query]['progress'] = 'No new data found'
            return

        # Create payload with data to validate (both new interactors and updates)
        new_only_payload = deepcopy(new_pipeline_payload)
        new_only_payload["ctx_json"]["interactors"] = interactors_to_validate

        # --- STAGE 2: Validate ONLY new data ---
        if not VALIDATOR_AVAILABLE:
            validated_new_payload = new_only_payload
        else:
            current_step += 1
            update_status(
                text="Validating new evidence...",
                current_step=current_step,
                total_steps=total_steps
            )

            validated_new_payload = validate_and_enrich_evidence(
                new_only_payload,
                api_key,
                verbose=False
            )

        # --- STAGE 2.12: Generate interaction metadata for new data ---
        if METADATA_GENERATOR_AVAILABLE and generate_interaction_metadata is not None:
            current_step += 1
            update_status(
                text="Analyzing interaction patterns...",
                current_step=current_step,
                total_steps=total_steps
            )
            validated_new_payload = generate_interaction_metadata(
                validated_new_payload,
                verbose=False
            )

        # --- STAGE 2.15: Update PMIDs for new data (before fact checker) ---
        if PMID_UPDATER_AVAILABLE and update_payload_pmids is not None:
            current_step += 1
            update_status(
                text="Validating citations...",
                current_step=current_step,
                total_steps=total_steps
            )
            validated_new_payload = update_payload_pmids(
                validated_new_payload,
                verbose=False
            )

        # --- STAGE 2.25: Deduplicate ONLY new data ---
        if skip_deduplicator:
            # Skip deduplication as requested by user (no progress update needed)
            pass
        elif DEDUPLICATOR_AVAILABLE and deduplicate_payload is not None:
            current_step += 1
            update_status(
                text="Deduplicating functions...",
                current_step=current_step,
                total_steps=total_steps
            )

            validated_new_payload = deduplicate_payload(
                validated_new_payload,
                api_key,
                verbose=False
            )

        # --- STAGE 2.5: Fact-check ONLY new data ---
        if not skip_fact_checking and FACT_CHECKER_AVAILABLE:
            current_step += 1
            update_status(
                text="Fact-checking new claims...",
                current_step=current_step,
                total_steps=total_steps
            )

            validated_new_payload = fact_check_json(
                validated_new_payload,
                api_key,
                verbose=False
            )

        # --- STAGE 3: Merge validated new data with existing ---
        current_step += 1
        update_status(
            text="Merging new results with existing data...",
            current_step=current_step,
            total_steps=total_steps
        )

        # Get validated new interactors
        validated_new_interactors = validated_new_payload.get("ctx_json", {}).get("interactors", [])

        # Post-processing: Remove duplicate functions before merging
        print(f"Re-query: Checking for duplicate functions before merge...", file=sys.stderr)
        deduplicated_new_interactors = []

        for new_int in validated_new_interactors:
            primary = new_int.get("primary")
            new_functions = new_int.get("functions", [])

            if not primary or not new_functions:
                deduplicated_new_interactors.append(new_int)
                continue

            # Get existing functions for this protein
            existing_funcs = existing_function_history.get(primary, [])

            # Filter out duplicate functions
            unique_functions = []
            duplicates_found = 0

            for func in new_functions:
                func_name = func.get("function", "").strip().lower()

                # Check if this function name already exists (case-insensitive)
                is_duplicate = any(
                    func_name == existing_func.strip().lower()
                    for existing_func in existing_funcs
                )

                if not is_duplicate:
                    unique_functions.append(func)
                else:
                    duplicates_found += 1
                    print(f"Re-query: Removed duplicate function '{func.get('function')}' for {primary}", file=sys.stderr)

            # Update interactor with only unique functions
            new_int_copy = deepcopy(new_int)
            new_int_copy["functions"] = unique_functions

            if unique_functions or primary not in existing_symbols:
                # Keep this interactor if it has unique functions OR is a new interactor
                deduplicated_new_interactors.append(new_int_copy)
            else:
                print(f"Re-query: Skipping {primary} - all functions were duplicates", file=sys.stderr)

        print(f"Re-query: Deduplication complete. Kept {len(deduplicated_new_interactors)} interactors", file=sys.stderr)

        # Merge with existing using deep merge
        merged_interactors = deep_merge_interactors(existing_interactors, deduplicated_new_interactors)

        # Update existing payload with merged data
        existing_ctx["interactors"] = merged_interactors

        # Update tracking lists
        existing_interactor_history = existing_ctx.get("interactor_history", [])
        new_interactor_history = validated_new_payload.get("ctx_json", {}).get("interactor_history", [])
        existing_ctx["interactor_history"] = existing_interactor_history + [
            x for x in new_interactor_history if x not in existing_interactor_history
        ]

        existing_function_batches = existing_ctx.get("function_batches", [])
        new_function_batches = validated_new_payload.get("ctx_json", {}).get("function_batches", [])
        existing_ctx["function_batches"] = existing_function_batches + [
            x for x in new_function_batches if x not in existing_function_batches
        ]

        # Merge function_history
        existing_func_hist = existing_ctx.get("function_history", {})
        new_func_hist = validated_new_payload.get("ctx_json", {}).get("function_history", {})
        for protein, funcs in new_func_hist.items():
            if protein in existing_func_hist:
                existing_func_hist[protein].extend(funcs)
            else:
                existing_func_hist[protein] = funcs
        existing_ctx["function_history"] = existing_func_hist

        # Rebuild snapshot with merged data
        merged_payload = deepcopy(existing_payload)
        merged_payload["ctx_json"] = existing_ctx

        # Regenerate snapshot_json from merged ctx_json
        merged_payload["snapshot_json"] = {
            "main": existing_ctx.get("main", user_query),
            "interactors": [
                {
                    "primary": i.get("primary"),
                    "direction": i.get("direction"),
                    "arrow": i.get("arrow"),
                    "intent": i.get("intent"),
                    "confidence": i.get("confidence"),
                    "support_summary": i.get("support_summary"),
                    "pmids": i.get("pmids", []),
                    "evidence": i.get("evidence", []),
                    "functions": i.get("functions", [])
                }
                for i in merged_interactors
            ]
        }

        # --- STAGE 4: Save merged results (split into 2 files) ---
        current_step += 1
        update_status(
            text="Caching results...",
            current_step=current_step,
            total_steps=total_steps
        )
        # File 1: PROTEIN.json - snapshot_json only (for visualization)
        output_path = os.path.join(CACHE_DIR, f"{user_query}.json")
        snapshot_only = {
            "snapshot_json": merged_payload.get("snapshot_json", {})
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_only, f, ensure_ascii=False, indent=2)

        # File 2: PROTEIN_metadata.json - ctx_json (full rich metadata)
        metadata_path = os.path.join(CACHE_DIR, f"{user_query}_metadata.json")
        metadata_only = {
            "ctx_json": merged_payload.get("ctx_json", {})
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_only, f, ensure_ascii=False, indent=2)

        # --- STAGE 4.5: Sync to PostgreSQL database ---
        from datetime import datetime
        import time

        sync_start_time = datetime.now()
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[RE-QUERY DB SYNC] Starting PostgreSQL sync for '{user_query}' at {sync_start_time.strftime('%H:%M:%S')}", file=sys.stderr)
        print(f"[RE-QUERY DB SYNC] Flask app provided: {flask_app is not None}", file=sys.stderr)

        if flask_app is None:
            print(f"[ERROR][RE-QUERY DB SYNC] Flask app instance is None - CANNOT sync to PostgreSQL!", file=sys.stderr)
            print(f"   Data preserved in file cache only", file=sys.stderr)
            print(f"   Run 'python sync_cache_to_db.py {user_query}' to sync manually", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        else:
            max_retries = 3
            retry_count = 0
            sync_success = False

            while retry_count < max_retries and not sync_success:
                try:
                    from utils.db_sync import DatabaseSyncLayer

                    if retry_count > 0:
                        wait_time = retry_count * 5
                        print(f"[RE-QUERY DB SYNC] Retry {retry_count}/{max_retries} in {wait_time}s...", file=sys.stderr)
                        time.sleep(wait_time)

                    # Test connection first
                    with flask_app.app_context():
                        from models import db
                        db.session.execute(db.text('SELECT 1'))
                        print(f"[RE-QUERY DB SYNC] [OK]Database connection verified", file=sys.stderr)

                    # CRITICAL: Flask-SQLAlchemy requires app context in background threads
                    with flask_app.app_context():
                        sync_layer = DatabaseSyncLayer()

                        num_interactions = len(merged_payload.get("snapshot_json", {}).get("interactors", []))
                        print(f"[RE-QUERY DB SYNC] Syncing protein '{user_query}' with {num_interactions} interactions...", file=sys.stderr)

                        db_stats = sync_layer.sync_query_results(
                            protein_symbol=user_query,
                            snapshot_json={"snapshot_json": merged_payload.get("snapshot_json", {})},
                            ctx_json=merged_payload.get("ctx_json")
                        )

                        sync_duration = (datetime.now() - sync_start_time).total_seconds()
                        print(f"[RE-QUERY DB SYNC] [OK]SUCCESS in {sync_duration:.1f}s", file=sys.stderr)
                        print(f"[RE-QUERY DB SYNC]   • Protein: {user_query}", file=sys.stderr)
                        print(f"[RE-QUERY DB SYNC]   • Interactions created: {db_stats['interactions_created']}", file=sys.stderr)
                        print(f"[RE-QUERY DB SYNC]   • Interactions updated: {db_stats['interactions_updated']}", file=sys.stderr)
                        print(f"{'='*60}\n", file=sys.stderr)
                        sync_success = True

                except Exception as db_error:
                    retry_count += 1
                    if retry_count >= max_retries:
                        # Final failure - log details and continue with file cache
                        print(f"\n[ERROR][RE-QUERY DB SYNC] FAILED after {max_retries} attempts", file=sys.stderr)
                        print(f"   Error: {db_error}", file=sys.stderr)
                        print(f"   Data preserved in file cache at: cache/{user_query}.json", file=sys.stderr)
                        print(f"   Run 'python sync_cache_to_db.py {user_query}' to sync manually", file=sys.stderr)
                        print(f"{'='*60}\n", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    else:
                        print(f"[WARN] [RE-QUERY DB SYNC] Attempt {retry_count} failed: {db_error}", file=sys.stderr)

        # Build detailed completion message with list of new items
        result_parts = []
        detailed_new_items = []

        if truly_new_interactors:
            result_parts.append(f"{len(truly_new_interactors)} new interactor{'s' if len(truly_new_interactors) != 1 else ''}")
            new_interactor_names = [i.get("primary", "Unknown") for i in truly_new_interactors]
            detailed_new_items.append(f"New interactors: {', '.join(new_interactor_names)}")

        if updated_existing_interactors:
            result_parts.append(f"{len(updated_existing_interactors)} updated interactor{'s' if len(updated_existing_interactors) != 1 else ''}")
            # Count new functions added to existing interactors
            for interactor in updated_existing_interactors:
                primary = interactor.get("primary", "Unknown")
                new_funcs = interactor.get("functions", [])
                if new_funcs:
                    func_names = [f.get("function", "Unknown") for f in new_funcs]
                    detailed_new_items.append(f"New functions for {primary}: {', '.join(func_names)}")

        result_message = "Added: " + ", ".join(result_parts) if result_parts else "No new data found"

        # Add detailed breakdown
        if detailed_new_items:
            result_message += " || " + " | ".join(detailed_new_items)

        print(f"Re-query: {result_message}", file=sys.stderr)
        print(f"Re-query: Saved to {output_path}", file=sys.stderr)

        # --- STAGE 5: Mark job as complete ---
        print(f"Re-query: Marking job as complete for {user_query}", file=sys.stderr)
        with lock:
            if user_query in jobs:
                if jobs[user_query].get('cancel_event') is cancel_event:
                    jobs[user_query]['status'] = 'complete'
                    jobs[user_query]['progress'] = result_message
                    print(f"Re-query: Successfully set status to 'complete'", file=sys.stderr)
                else:
                    print(f"Re-query: Cancel event mismatch, not updating status", file=sys.stderr)
            else:
                print(f"Re-query: Job {user_query} not found in jobs dict", file=sys.stderr)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        is_cancelled = "cancelled by user" in error_message.lower()

        if is_cancelled:
            print(f"RE-QUERY CANCELLED for '{user_query}'", file=sys.stderr)
            with lock:
                if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                    jobs[user_query]['status'] = 'cancelled'
                    jobs[user_query]['progress'] = {"text": "Cancelled by user"}
        else:
            print(f"RE-QUERY ERROR for '{user_query}': {error_message}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

            with lock:
                if user_query in jobs and jobs[user_query].get('cancel_event') is cancel_event:
                    jobs[user_query]['status'] = 'error'
                    jobs[user_query]['progress'] = {"text": error_message}


# ============================================================================
# CLI INTERFACE (ORIGINAL FUNCTIONALITY PRESERVED)
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced pipeline runner with evidence validation"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Protein to analyze"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debugging info"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming previews"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path (default: <query>_pipeline.json)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--viz-only",
        type=str,
        help="Only generate visualization from existing JSON"
    )
    parser.add_argument(
        "--validate-evidence",
        action="store_true",
        help="Run evidence validator after pipeline (RECOMMENDED)"
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=3,
        help="Batch size for evidence validation (default: 3)"
    )
    parser.add_argument(
        "--interactor-rounds",
        type=int,
        help="Number of interactor discovery rounds (default: 3, min: 3, max: 10)"
    )
    parser.add_argument(
        "--function-rounds",
        type=int,
        help="Number of function mapping rounds (default: 3, min: 3, max: 10)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for number of discovery rounds interactively"
    )

    args = parser.parse_args()

    # Viz-only mode
    if args.viz_only:
        json_file = Path(args.viz_only)
        if not json_file.exists():
            parser.error(f"JSON file not found: {json_file}")

        print(f"Creating visualization from: {json_file}")
        html_file = create_visualization(json_file)
        print("Visualization created!")
        return

    # Normal pipeline mode
    if not args.query:
        try:
            args.query = input("Enter protein name: ").strip()
        except EOFError:
            args.query = ""

    if not args.query:
        parser.error("query is required")

    ensure_env()

    # Determine number of rounds (interactive or from args)
    num_interactor_rounds = 3  # Default
    num_function_rounds = 3    # Default

    if args.interactive or (not args.interactor_rounds and not args.function_rounds):
        # Interactive mode - prompt user
        print(f"\n{'='*80}")
        print("PIPELINE CONFIGURATION")
        print(f"{'='*80}")
        print("\nDefault configuration:")
        print("  - Interactor discovery rounds: 3 (1a, 1b, 1c)")
        print("  - Function mapping rounds: 3 (2a, 2a2, 2a3)")
        print("\nYou can customize the number of rounds for more comprehensive results.")
        print("More rounds = more interactors and functions discovered (but longer runtime)")
        print(f"{'='*80}\n")

        try:
            interactor_input = input("Number of interactor discovery rounds (3-10, default 3): ").strip()
            if interactor_input:
                num_interactor_rounds = int(interactor_input)
                num_interactor_rounds = max(3, min(10, num_interactor_rounds))

            function_input = input("Number of function mapping rounds (3-10, default 3): ").strip()
            if function_input:
                num_function_rounds = int(function_input)
                num_function_rounds = max(3, min(10, num_function_rounds))
        except (ValueError, EOFError):
            print("\nUsing defaults (3 rounds each)")

    # Override with command-line args if provided
    if args.interactor_rounds:
        num_interactor_rounds = max(3, min(10, args.interactor_rounds))
    if args.function_rounds:
        num_function_rounds = max(3, min(10, args.function_rounds))

    # Show configuration
    print(f"\n{'='*80}")
    print("RUNNING PIPELINE WITH:")
    print(f"{'='*80}")
    print(f"  Protein: {args.query}")
    print(f"  Interactor discovery rounds: {num_interactor_rounds}")
    print(f"  Function mapping rounds: {num_function_rounds}")
    if DYNAMIC_CONFIG_AVAILABLE:
        print(f"  Dynamic configuration: ENABLED")
    else:
        print(f"  Dynamic configuration: NOT AVAILABLE (using defaults)")
    print(f"{'='*80}\n")

    # Run main pipeline
    final_payload, step_logger = run_pipeline(
        user_query=args.query,
        verbose=args.verbose,
        stream=not args.no_stream,
        num_interactor_rounds=num_interactor_rounds,
        num_function_rounds=num_function_rounds,
    )

    # Save initial output
    output_path = Path(args.output) if args.output else Path(f"{args.query}_pipeline.json")
    output_path.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[OK]Saved pipeline output to: {output_path}")

    # Save NDJSON if present
    ndjson_content = final_payload.get("ndjson")
    if ndjson_content:
        ndjson_path = output_path.with_suffix(".ndjson")
        if isinstance(ndjson_content, list):
            ndjson_text = "\n".join(
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in ndjson_content
            )
        else:
            ndjson_text = str(ndjson_content)
        ndjson_path.write_text(ndjson_text.rstrip() + "\n", encoding="utf-8")
        print(f"[OK]Saved NDJSON to: {ndjson_path}")

    # Evidence validation (if requested)
    if args.validate_evidence:
        if not VALIDATOR_AVAILABLE:
            print("\n[WARN]Evidence validator not available. Skipping validation.")
        else:
            print(f"\n{'='*80}")
            print("RUNNING EVIDENCE VALIDATION")
            print(f"{'='*80}")

            api_key = os.getenv("GOOGLE_API_KEY")
            validated_payload = validate_and_enrich_evidence(
                final_payload,
                api_key,
                verbose=args.verbose,
                batch_size=args.validation_batch_size,
                step_logger=step_logger
            )

            # Save validated output
            validated_path = output_path.parent / f"{output_path.stem}_validated{output_path.suffix}"
            validated_path.write_text(
                json.dumps(validated_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8"
            )
            print(f"\n[OK]Saved validated output to: {validated_path}")

            # Use validated output for visualization
            output_path = validated_path
            final_payload = validated_payload

    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")

    if "ctx_json" in final_payload:
        interactors = final_payload["ctx_json"].get("interactors", [])
        total_functions = sum(len(i.get("functions", [])) for i in interactors)
        total_pmids = sum(len(i.get("pmids", [])) for i in interactors)

        print(f"[OK]Found {len(interactors)} interactors")
        print(f"[OK]Mapped {total_functions} biological functions")
        print(f"[OK]Collected {total_pmids} citations")

    # Generate visualization
    if not args.no_viz:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATION")
        print(f"{'='*80}")
        html_path = output_path.with_suffix(".html")
        viz_file = create_visualization(output_path, html_path)
        print(f"[OK]Visualization saved to: {viz_file}")


if __name__ == "__main__":
    main()
