#!/usr/bin/env python3
"""
Evidence Validator & Citation Enricher (MERGED WITH FACT-CHECKER)
Post-processes pipeline JSON to validate claims, correct mechanisms, and extract independent evidence.
Uses Gemini 3.0 Pro with Google Search for maximum accuracy and reasoning.
"""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import httpx
from google.genai import types
from dotenv import load_dotenv

# Constants
MAX_THINKING_TOKENS = 32768
MAX_OUTPUT_TOKENS = 65536
MIN_THINKING_BUDGET = 1000

class EvidenceValidatorError(RuntimeError):
    pass

def load_json_file(json_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise EvidenceValidatorError(f"Failed to load JSON: {e}")

def save_json_file(data: Dict[str, Any], output_path: Path) -> None:
    try:
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"[OK]Saved validated output to: {output_path}")
    except Exception as e:
        raise EvidenceValidatorError(f"Failed to save JSON: {e}")

def call_gemini_with_search(prompt: str, api_key: str, system_message: Optional[str] = None, verbose: bool = False) -> str:
    from google import genai as google_genai
    client = google_genai.Client(api_key=api_key)

    # Use gemini-2.5-pro as requested (or best available)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,
            include_thoughts=True,
        ),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        top_p=0.90,
    )

    if verbose:
        print(f"CALLING GEMINI WITH SEARCH (Thinking: {MAX_THINKING_TOKENS})")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            full_prompt = prompt
            if system_message:
                full_prompt = system_message + "\n\n" + prompt

            # User requested 3.0, defaulting to 2.5-pro as stable alias
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=full_prompt,
                config=config,
            )

            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                return ''.join(part.text for part in parts if hasattr(part, 'text')).strip()
            raise EvidenceValidatorError("No text in response")
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
            else:
                raise EvidenceValidatorError(f"Failed after {max_retries} attempts: {e}")

def extract_json_from_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Simple fallback extraction
        try:
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            return json.loads(cleaned[start:end])
        except:
            raise EvidenceValidatorError(f"Failed to parse JSON response")

def validate_and_enrich_evidence(
    json_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False,
    batch_size: int = 3,
    step_logger=None
) -> Dict[str, Any]:
    if 'ctx_json' not in json_data:
        raise EvidenceValidatorError("No ctx_json found")

    ctx_json = json_data['ctx_json']
    interactors = ctx_json.get('interactors', [])
    main_protein = ctx_json.get('main', 'UNKNOWN')
    
    print(f"VALIDATING & CORRECTING {len(interactors)} INTERACTORS FOR {main_protein}")
    
    validated_interactors = []
    for batch_start in range(0, len(interactors), batch_size):
        batch_end = min(batch_start + batch_size, len(interactors))
        batch = interactors[batch_start:batch_end]
        print(f"Processing batch {batch_start}-{batch_end}")

        prompt = create_validation_prompt(main_protein, batch, batch_start, batch_end, len(interactors))
        try:
            resp = call_gemini_with_search(prompt, api_key, verbose=verbose)
            val_data = extract_json_from_response(resp)

            if 'interactors' in val_data:
                # Merge logic here
                orig_map = {i.get('primary'): i for i in batch}
                for val_int in val_data['interactors']:
                    if val_int.get('_delete_interactor'):
                        print(f"  [DELETE] {val_int.get('primary')}")
                        continue

                    orig = orig_map.get(val_int.get('primary'))
                    if orig:
                        # Preserve metadata
                        for k, v in orig.items():
                            if k.startswith('_') and k not in val_int:
                                val_int[k] = v
                        validated_interactors.append(val_int)
            else:
                validated_interactors.extend(batch)

        except Exception as e:
            print(f"Batch failed: {e}")
            validated_interactors.extend(batch)

    ctx_json['interactors'] = validated_interactors
    if 'snapshot_json' in json_data:
        json_data['snapshot_json']['interactors'] = validated_interactors

    # Format cascades
    json_data = format_biological_cascades(json_data, api_key, verbose, step_logger)
    return json_data

def create_validation_prompt(
    main_protein: str,
    batch: List[Dict[str, Any]],
    batch_start: int,
    batch_end: int,
    total_count: int
) -> str:
    """Create a detailed validation prompt for Gemini using strict fact-checking logic."""
    
    batch_json = json.dumps(batch, indent=2, ensure_ascii=False)
    
    prompt = f"""STRICT FACT-CHECKING & CORRECTION TASK

You are a RIGOROUS scientific fact-checker. Your job is to VALIDATE claims, CORRECT errors, and DELETE fabrications.
You are reviewing protein-protein interactions for {main_protein}.

PROCESSING: Interactors {batch_start+1}-{batch_end} of {total_count}

INPUT DATA (JSON):
{batch_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY RESEARCH PROTOCOL (DO NOT SKIP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **INDEPENDENT RESEARCH**: The input data has NO evidence. You MUST search PubMed/Google Scholar from scratch.
   - Search: "{main_protein} [Interactor] interaction"
   - Search: "{main_protein} [Interactor] [Function]"
   - Find 3-5 real papers for each interaction.

2. **RED FLAG DETECTION**:
   - **Co-localization ≠ Interaction**: "Both in stress granules" is NOT a functional interaction. → DELETE
   - **Wrong Mechanism**: Claim says "stabilizes" but papers show "degrades". → CORRECT
   - **Wrong Protein**: Claim says "ATXN3-HDAC1" but papers show "ATXN3-HDAC3". → DELETE
   - **Wrong Direction**: Claim says "activates" but papers show "inhibits". → CORRECT

3. **CORRECTION LOGIC**:
   - If the core interaction is REAL but the function/mechanism is WRONG: **REWRITE IT**.
   - Example: Claim "ATXN3 stabilizes PTEN" vs Reality "ATXN3 represses PTEN transcription".
     → **ACTION**: Change function to "Transcriptional Repression", arrow to "inhibits", mechanism to "promoter binding".
   - If the interaction is FAKE or just CO-LOCALIZATION: **DELETE IT**.
     → Set `_delete_interactor: true`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. **EVIDENCE REQUIREMENTS**:
   - For every TRUE or CORRECTED function, you must provide independent evidence.
   - **Paper Title**: EXACT word-for-word title from PubMed (REQUIRED).
   - **Quote**: <=200 char quote from the paper showing the interaction mechanism (REQUIRED).
   - **Metadata**: Authors, Year, Journal.

B. **FUNCTION BOX REFINEMENT**:
   Even for TRUE claims, you must REFINE the details:
   - **Arrow**: Ensure 'activates' vs 'inhibits' matches reality.
   - **Biological Consequence**: Add missing downstream steps (use canonical pathway knowledge).
   - **Specific Effects**: Remove unsupported effects, add supported ones.

C. **BIOLOGICAL CASCADES**:
   - Format as: ["Event A", "Event B", "Outcome C"]
   - Use inference to complete chains (e.g., if A degrades B, and B inhibits C, then A activates C).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON matching this structure.

{{
  "interactors": [
    {{
      "primary": "GENE_SYMBOL",
      "_delete_interactor": false, // Set true if fabrication/co-localization only
      "validation_note": "Explanation of validation/correction",
      "direction": "main_to_primary|primary_to_main|bidirectional",
      "arrow": "activates|inhibits|binds|regulates",
      "intent": "mechanism (e.g., deubiquitination)",
      "functions": [
        {{
          "function": "CORRECTED Function Name",
          "arrow": "activates|inhibits",
          "cellular_process": "Corrected description of mechanism",
          "effect_description": "Corrected summary of outcome",
          "biological_consequence": ["Step 1", "Step 2", "Step 3"],
          "specific_effects": ["Effect 1", "Effect 2"],
          "evidence": [
            {{
              "paper_title": "EXACT TITLE FROM PUBMED",
              "relevant_quote": "Direct quote showing mechanism",
              "doi": "...",
              "authors": "...",
              "year": 2024,
              "journal": "..."
            }}
          ]
        }}
      ]
    }}
  ]
}}

Begin validation. Be aggressive in correcting errors.
"""

    return prompt


def format_biological_cascades(
    json_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False,
    step_logger=None
) -> Dict[str, Any]:
    """
    Format and validate biological cascades using Gemini.
    Ensures cascades are logically structured, scientifically accurate, and beautifully formatted.
    """
    # Log step start
    if step_logger:
        step_logger.log_step_start(
            step_name="cascade_formatting",
            input_data=json_data,
            step_type="post_processing"
        )

    # Work on ctx_json
    if 'ctx_json' not in json_data:
        return json_data

    ctx = json_data['ctx_json']
    interactors = ctx.get('interactors', [])

    if not interactors:
        return json_data

    main_protein = ctx.get('main', 'UNKNOWN')

    # Collect all functions that need cascade formatting
    functions_to_format = []
    for idx, interactor in enumerate(interactors):
        primary = interactor.get('primary', '')
        functions = interactor.get('functions', [])

        for fidx, function in enumerate(functions):
            bio_cascades = function.get('biological_consequence', [])
            if not bio_cascades or not any(bio_cascades):  # Skip empty or None
                continue

            functions_to_format.append({
                'interactor_idx': idx,
                'function_idx': fidx,
                'primary': primary,
                'function_name': function.get('function', ''),
                'cellular_process': function.get('cellular_process', ''),
                'specific_effects': function.get('specific_effects', []),
                'current_cascades': bio_cascades,
                'arrow': function.get('arrow', '')
            })

    if not functions_to_format:
        return json_data

    print(f"  Found {len(functions_to_format)} function(s) with biological cascades")

    # Process in batches to manage token usage
    batch_size = 5  # Process 5 functions at a time
    total_batches = (len(functions_to_format) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(functions_to_format))
        batch = functions_to_format[start_idx:end_idx]

        print(f"\n  Batch {batch_num + 1}/{total_batches} ({len(batch)} functions)")

        # Build prompt for this batch
        prompt = create_cascade_formatting_prompt(main_protein, batch)

        try:
            # Call Gemini with same config as evidence validator
            response_text = call_gemini_with_search(prompt, api_key, verbose=verbose)

            # Parse response
            formatted_data = extract_json_from_response(response_text)

            # Apply formatted cascades back to json_data
            if 'functions' in formatted_data:
                for i, formatted_fn in enumerate(formatted_data['functions']):
                    if i < len(batch):
                        func_ref = batch[i]
                        interactor_idx = func_ref['interactor_idx']
                        function_idx = func_ref['function_idx']

                        # Get the formatted cascades
                        new_cascades = formatted_fn.get('biological_consequence', [])

                        if new_cascades:
                            # Update in ctx_json
                            interactors[interactor_idx]['functions'][function_idx]['biological_consequence'] = new_cascades

        except Exception as e:
            print(f"  [WARN]Cascade formatting failed for batch {batch_num + 1}: {e}")

    # Also update snapshot_json if it exists
    if 'snapshot_json' in json_data:
        json_data['snapshot_json']['interactors'] = interactors

    print(f"\n  [OK]Cascade formatting complete!")
    return json_data


def create_cascade_formatting_prompt(main_protein: str, functions: list) -> str:
    """
    Create a prompt for Gemini to format biological cascades.
    """
    prompt = f"""TASK: Format and validate biological cascades for scientific clarity and accuracy

You are a molecular biology expert reviewing biological cascade descriptions.

MAIN PROTEIN: {main_protein}

FORMATTING REQUIREMENTS:
1. **Logical Flow**: Clear cause → effect progression.
2. **Arrow Notation**: Use consistent arrows (→) to separate steps.
3. **Scientific Accuracy**: Verify steps against canonical pathways.
4. **Consistency**: Ensure steps match the function name.

FUNCTIONS TO FORMAT:
"""

    for i, fn in enumerate(functions, 1):
        prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTION {i}: {fn['primary']} → {fn['function_name']}
Cellular Process: {fn['cellular_process']}
Current Cascades: {fn['current_cascades']}
"""

    prompt += """
OUTPUT FORMAT (JSON):
{
  "functions": [
    {
      "biological_consequence": [
        "Step 1 → Step 2 → Outcome"
      ]
    }
  ]
}
"""
    return prompt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return
    
    data = load_json_file(Path(args.input_json))
    val_data = validate_and_enrich_evidence(data, api_key, args.verbose, args.batch_size)
    
    out = Path(args.output) if args.output else Path(args.input_json).with_suffix(".validated.json")
    save_json_file(val_data, out)

if __name__ == "__main__":
    main()
