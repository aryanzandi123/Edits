#!/usr/bin/env python3
"""
Evidence Validator & Citation Enricher
Post-processes pipeline JSON to validate PMIDs, extract quotes, and ensure unique evidence per function.
Uses Gemini 2.5 Pro with Google Search for maximum accuracy.
"""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Fix Windows console encoding for Greek letters and special characters
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import httpx
from google.genai import types
from dotenv import load_dotenv

# Constants for Gemini API
MAX_THINKING_TOKENS = 32768  # Updated to match claim_fact_checker.py
MAX_OUTPUT_TOKENS = 65536
MIN_THINKING_BUDGET = 1000

class EvidenceValidatorError(RuntimeError):
    """Raised when evidence validation fails."""
    pass


def load_json_file(json_path: Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise EvidenceValidatorError(f"Failed to load JSON: {e}")


def save_json_file(data: Dict[str, Any], output_path: Path) -> None:
    """Save data to JSON file with pretty formatting."""
    try:
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"[OK]Saved validated output to: {output_path}")
    except Exception as e:
        raise EvidenceValidatorError(f"Failed to save JSON: {e}")


def call_gemini_with_search(
    prompt: str,
    api_key: str,
    system_message: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Call Gemini 2.5 Pro with maximum thinking budget and Google Search enabled.

    Args:
        prompt: User prompt text
        api_key: Google API key
        system_message: Optional system message (prepended to contents)
        verbose: Print detailed output

    Returns:
        Model response text
    """
    from google import genai as google_genai

    client = google_genai.Client(api_key=api_key)

    # Configure with maximum capabilities
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,
            include_thoughts=True,
        ),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,  # Lower temperature for factual accuracy
        top_p=0.90,
    )

    if verbose:
        print(f"\n{'='*80}")
        print("CALLING GEMINI 2.5 PRO WITH EVIDENCE VALIDATION")
        print(f"{'='*80}")
        print(f"Thinking Budget: {MAX_THINKING_TOKENS:,} tokens")
        print(f"Output Limit: {MAX_OUTPUT_TOKENS:,} tokens")
        print(f"Google Search: ENABLED")
        if system_message:
            print(f"System Message: {len(system_message)} chars")

    max_retries = 5
    base_delay = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            if verbose and attempt > 1:
                print(f"\nRetry attempt {attempt}/{max_retries}...")

            # Prepend system message to prompt if provided (same pattern as claim_fact_checker)
            full_prompt = prompt
            if system_message:
                full_prompt = system_message + "\n\n" + prompt

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=full_prompt,
                config=config,
            )

            # Extract text from response
            if hasattr(response, 'text'):
                output = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                output = ''.join(part.text for part in parts if hasattr(part, 'text'))
            else:
                raise EvidenceValidatorError("No text in response")

            # Extract token usage
            usage = getattr(response, 'usage_metadata', None)
            token_stats = {'thinking': 0, 'output': 0, 'total': 0, 'input': 0}
            if usage:
                thinking_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                total_tokens = getattr(usage, 'total_token_count', 0) or 0
                prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0

                # Calculate thinking tokens if not provided
                if total_tokens > 0 and thinking_tokens == 0:
                    thinking_tokens = max(0, total_tokens - prompt_tokens - output_tokens)

                input_tokens = max(0, total_tokens - thinking_tokens - output_tokens)

                token_stats = {
                    'thinking': thinking_tokens,
                    'output': output_tokens,
                    'total': total_tokens,
                    'input': input_tokens
                }

                # Calculate costs (Gemini 2.5 Pro pricing)
                input_cost = (input_tokens / 1_000_000) * 1.25
                thinking_cost = (thinking_tokens / 1_000_000) * 1.25
                output_cost = (output_tokens / 1_000_000) * 10.00
                total_cost = input_cost + thinking_cost + output_cost

                print(f"    → Tokens: input={input_tokens:,}, thinking={thinking_tokens:,}, output={output_tokens:,}, total={total_tokens:,}")
                print(f"    → Cost: ${total_cost:.4f} (input: ${input_cost:.4f}, thinking: ${thinking_cost:.4f}, output: ${output_cost:.4f})")

            if verbose:
                print(f"\n[OK]Response received ({len(output)} chars)")

            return output.strip()
            
        except Exception as e:
            error_msg = str(e)
            delay = base_delay * (2 ** (attempt - 1))
            
            if attempt < max_retries:
                print(f"[WARN]Error: {error_msg}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise EvidenceValidatorError(f"Failed after {max_retries} attempts: {error_msg}")


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extract JSON from model response, handling markdown fences."""
    cleaned = text.strip()
    
    # Remove markdown code fences
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to find JSON in the text
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            try:
                return json.loads(cleaned[start_idx:end_idx])
            except:
                pass
        raise EvidenceValidatorError(f"Failed to parse JSON: {e}\nText: {cleaned[:500]}")


def merge_preserving_validated_arrows(
    original_interactor: Dict[str, Any],
    validated_interactor: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge validated interactor data while preserving arrow fields that have been validated.

    If the original interactor has _arrow_validated=True, preserve arrow, direction,
    and interaction_type fields from the original.

    Args:
        original_interactor: Original interactor with potential _arrow_validated flag
        validated_interactor: Updated interactor from Gemini validation

    Returns:
        Merged interactor with protected arrow fields
    """
    # Check if arrows were validated
    if original_interactor.get('_arrow_validated'):
        # Preserve validated arrow fields
        validated_interactor['arrow'] = original_interactor.get('arrow')
        validated_interactor['direction'] = original_interactor.get('direction')
        validated_interactor['interaction_type'] = original_interactor.get('interaction_type')

        # Preserve validation metadata
        validated_interactor['_arrow_validated'] = True
        if '_validation_metadata' in original_interactor:
            validated_interactor['_validation_metadata'] = original_interactor['_validation_metadata']

        # Also preserve function arrows if they were validated
        original_functions = original_interactor.get('functions', [])
        validated_functions = validated_interactor.get('functions', [])

        # Match functions by name and preserve arrows
        for orig_func in original_functions:
            func_name = orig_func.get('function', '')
            for val_func in validated_functions:
                if val_func.get('function', '') == func_name:
                    # Preserve validated function arrow
                    val_func['arrow'] = orig_func.get('arrow')
                    break

    return validated_interactor


def validate_and_enrich_evidence(
    json_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False,
    batch_size: int = 3,
    step_logger=None
) -> Dict[str, Any]:
    """
    Validate and enrich evidence for all interactors and functions.
    Processes in batches to avoid overwhelming the model.

    Args:
        json_data: Pipeline output JSON with ctx_json
        api_key: Google API key for Gemini
        verbose: Print detailed debugging info
        batch_size: Number of interactors to process per batch
        step_logger: Optional StepLogger instance for comprehensive logging
    """
    # Log step start
    if step_logger:
        step_logger.log_step_start(
            step_name="evidence_validation",
            input_data=json_data,
            step_type="post_processing"
        )

    # Work on ctx_json which has the full data
    if 'ctx_json' not in json_data:
        raise EvidenceValidatorError("No ctx_json found in input data")

    ctx_json = json_data['ctx_json']
    interactors = ctx_json.get('interactors', [])

    if not interactors:
        print("[WARN]No interactors found to validate")
        if step_logger:
            step_logger.log_terminal_output("[WARN]No interactors found to validate")
        return json_data

    main_protein = ctx_json.get('main', 'UNKNOWN')
    print(f"\n{'='*80}")
    print(f"VALIDATING EVIDENCE FOR {main_protein}")
    print(f"{'='*80}")
    print(f"Total interactors to process: {len(interactors)}")

    # Capture terminal output
    if step_logger:
        step_logger.log_terminal_output(f"{'='*80}")
        step_logger.log_terminal_output(f"VALIDATING EVIDENCE FOR {main_protein}")
        step_logger.log_terminal_output(f"{'='*80}")
        step_logger.log_terminal_output(f"Total interactors to process: {len(interactors)}")
    
    # Process interactors in batches
    validated_interactors = []
    
    for batch_start in range(0, len(interactors), batch_size):
        batch_end = min(batch_start + batch_size, len(interactors))
        batch = interactors[batch_start:batch_end]

        batch_start_time = time.time()
        print(f"\n--- Processing batch {batch_start//batch_size + 1} "
              f"(interactors {batch_start+1}-{batch_end}) ---")

        # Create validation prompt for this batch
        prompt = create_validation_prompt(main_protein, batch, batch_start, batch_end, len(interactors))

        # Call Gemini with search
        response_text = call_gemini_with_search(prompt, api_key, verbose)

        # Parse response
        try:
            validated_batch = extract_json_from_response(response_text)

            if 'interactors' in validated_batch:
                # Merge validated data while preserving arrow validation
                merged_batch = []
                for orig_interactor, val_interactor in zip(batch, validated_batch['interactors']):
                    merged = merge_preserving_validated_arrows(orig_interactor, val_interactor)
                    merged_batch.append(merged)

                validated_interactors.extend(merged_batch)
                print(f"[OK]Validated {len(merged_batch)} interactors in this batch")

                # Count how many had protected arrows
                protected_count = sum(1 for m in merged_batch if m.get('_arrow_validated'))
                if protected_count > 0:
                    print(f"    → Protected {protected_count} arrow(s) from arrow_effect_validator")
            else:
                print(f"[WARN]No interactors in response, keeping original batch")
                validated_interactors.extend(batch)

        except Exception as e:
            print(f"[WARN]Error parsing batch response: {e}")
            print("Keeping original batch")
            validated_interactors.extend(batch)

        batch_elapsed = time.time() - batch_start_time
        print(f"    [TIME] Batch time: {batch_elapsed:.1f}s")
    
    # Update the JSON data
    ctx_json['interactors'] = validated_interactors
    
    # Also update snapshot_json if it exists
    if 'snapshot_json' in json_data:
        json_data['snapshot_json']['interactors'] = validated_interactors
    
    print(f"\n[OK]Evidence validation complete!")
    print(f"  Total interactors processed: {len(validated_interactors)}")
    
    # Calculate statistics
    total_functions = sum(len(i.get('functions', [])) for i in validated_interactors)
    total_pmids = sum(len(i.get('pmids', [])) for i in validated_interactors)
    functions_with_evidence = sum(
        1 for i in validated_interactors 
        for f in i.get('functions', []) 
        if f.get('pmids') and f.get('evidence')
    )
    
    print(f"  Total functions: {total_functions}")
    print(f"  Total PMIDs: {total_pmids}")
    print(f"  Functions with evidence: {functions_with_evidence}")

    # Capture terminal output
    if step_logger:
        step_logger.log_terminal_output(f"\n[OK]Evidence validation complete!")
        step_logger.log_terminal_output(f"  Total interactors processed: {len(validated_interactors)}")
        step_logger.log_terminal_output(f"  Total functions: {total_functions}")
        step_logger.log_terminal_output(f"  Total PMIDs: {total_pmids}")
        step_logger.log_terminal_output(f"  Functions with evidence: {functions_with_evidence}")

    # Log step completion
    if step_logger:
        step_metadata = {
            'total_interactors': len(validated_interactors),
            'total_functions': total_functions,
            'total_pmids': total_pmids,
            'functions_with_evidence': functions_with_evidence
        }
        step_logger.log_step_complete(
            output_data=json_data,
            metadata=step_metadata,
            generate_summary=True
        )

    # Format biological cascades for clarity and consistency
    print(f"\n{'─'*80}")
    print("FORMATTING BIOLOGICAL CASCADES...")
    print(f"{'─'*80}")
    json_data = format_biological_cascades(json_data, api_key, verbose=verbose, step_logger=step_logger)

    return json_data


def create_validation_prompt(
    main_protein: str,
    batch: List[Dict[str, Any]],
    batch_start: int,
    batch_end: int,
    total_count: int
) -> str:
    """Create a detailed validation prompt for Gemini."""
    
    batch_json = json.dumps(batch, indent=2, ensure_ascii=False)
    
    prompt = f"""EVIDENCE VALIDATION & ENRICHMENT TASK

You are validating and enriching evidence for protein-protein interactions. This is CRITICAL for scientific accuracy.

MAIN PROTEIN: {main_protein}
PROCESSING: Interactors {batch_start+1}-{batch_end} of {total_count}

INPUT DATA (JSON):
{batch_json}

MANDATORY TASKS:

1. **VALIDATE PAPER TITLES (PRIMARY FOCUS)**:
   - **Paper titles are the PRIMARY evidence** - PMID validation is handled by update_cache_pmids.py
   - Use Google Search to verify EVERY paper title is EXACT and verifiable
   - Search: "[paper_title]" or "[paper_title] [authors]"
   - If paper title is invalid or doesn't exist, REMOVE the evidence entry
   - DO NOT validate PMIDs - that's handled by a separate tool
   - DO NOT remove evidence entries just because PMID is missing
   - Focus ONLY on ensuring paper titles are EXACT word-for-word from PubMed/Google Scholar

2. **FIX ARROW TYPES** (currently many are wrong):
   - Analyze the interaction mechanism from literature
   - Set arrow to:
     * 'activates': if one protein enhances/promotes the other's activity
     * 'inhibits': if one protein suppresses/reduces the other's activity  
     * 'binds': ONLY if pure binding with no functional consequence
   - Update 'intent' to match mechanism (phosphorylation, ubiquitination, etc.)
   - Set 'direction': 'main_to_primary' or 'primary_to_main' or 'bidirectional'

3. **EXTRACT EXACT QUOTES**:
   - For EACH function, find the primary paper
   - Extract a <=200 character quote that DIRECTLY supports the claim
   - Quote must mention BOTH proteins and the specific mechanism
   - Store in evidence[].relevant_quote

4. **ENSURE UNIQUE EVIDENCE PER FUNCTION**:
   - Each function box must have its OWN specific PAPER TITLES
   - Don't reuse the same paper title for all functions unless truly applicable
   - Add 2-5 PAPER TITLES per function from different papers if available
   - Update function.evidence arrays (PMIDs will be added by update_cache_pmids.py)

5. **EXPAND BIOLOGICAL CONSEQUENCES**:
   - If a function has multiple downstream effects, create MULTIPLE biological_consequence entries
   - Format as array of strings: biological_consequence: ["cascade 1", "cascade 2", ...]
   - Each cascade should be detailed and specific with arrows (→)
   - Include pathway names, affected processes, and cellular outcomes

6. **ADD EFFECT DESCRIPTION** (NEW FIELD):
   - For each function, add effect_description: one sentence describing what happens to the function
   - Example: "Histone acetyltransferase activity is inhibited, reducing chromatin acetylation"
   - This should summarize the functional outcome clearly

7. **MAINTAIN SPECIFIC_EFFECTS DISTINCTION**:
   - specific_effects should be direct, concise facts (NO arrows)
   - Example: "ATXN3 inhibits CBP acetyltransferase activity"
   - NOT: "X → Y → Z" (that goes in biological_consequence)

8. **COMPLETE METADATA**:
   - Ensure every evidence entry has: paper_title (REQUIRED), doi, authors, journal, year, assay, species
   - paper_title is the MOST IMPORTANT field - must be EXACT word-for-word
   - pmid is OPTIONAL - will be populated later by update_cache_pmids.py
   - Get metadata from PubMed/Google Scholar literature search
   - Never invent or hallucinate - if uncertain, search again

CRITICAL RULES:
- **Paper titles are PRIMARY evidence** (PMIDs handled separately by update_cache_pmids.py)
- Never invent paper titles, DOIs, or paper details
- If you can't verify a paper title, remove the entire evidence entry
- Use Google Search extensively - search for EVERY paper title and interaction
- Be thorough: spend thinking budget on finding the RIGHT papers with EXACT titles
- Focus on evidence quality, NOT PMID validation
- Each function must have unique, verifiable PAPER TITLES
- Extract real quotes from actual papers (use search to access papers)

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
  "interactors": [
    {{
      "primary": "GENE_SYMBOL",
      "direction": "main_to_primary|primary_to_main|bidirectional",
      "arrow": "activates|inhibits|binds",
      "intent": "specific_mechanism",
      "pmids": ["########", ...],
      "evidence": [
        {{
          "pmid": "########",
          "doi": "10.xxxx/xxxxx",
          "paper_title": "Exact title from PubMed",
          "authors": "LastName1 A, LastName2 B",
          "journal": "Journal Name",
          "year": 2024,
          "assay": "Co-IP|Western|etc",
          "species": "human|mouse|rat",
          "relevant_quote": "<=200 char excerpt mentioning both proteins and mechanism"
        }}
      ],
      "support_summary": "Brief summary",
      "functions": [
        {{
          "function": "Short function name",
          "arrow": "activates|inhibits",
          "cellular_process": "Detailed mechanism",
          "effect_description": "One sentence describing what happens to the function",
          "biological_consequence": ["Cascade 1: pathway → step → outcome", "Cascade 2: different pathway"],
          "specific_effects": ["Direct fact 1 (NO arrows)", "Direct fact 2 (NO arrows)"],
          "pmids": ["########", "########"],
          "evidence": [
            {{
              "pmid": "########",
              "doi": "...",
              "paper_title": "...",
              "authors": "...",
              "journal": "...",
              "year": 2023,
              "assay": "...",
              "species": "...",
              "relevant_quote": "Direct quote supporting THIS specific function"
            }}
          ],
          "mechanism_id": "optional_link"
        }}
      ]
    }}
  ]
}}

Begin validation. Use extensive Google Search to verify every claim."""

    return prompt


def format_biological_cascades(
    json_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False,
    step_logger=None
) -> Dict[str, Any]:
    """
    Format and validate biological cascades using Gemini 2.5 Pro.
    Ensures cascades are logically structured, scientifically accurate, and beautifully formatted.

    Args:
        json_data: Pipeline output JSON with ctx_json
        api_key: Google API key for Gemini
        verbose: Print detailed debugging info
        step_logger: Optional StepLogger instance for comprehensive logging
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
        if verbose:
            print("[WARN]No ctx_json found, skipping cascade formatting")
        return json_data

    ctx = json_data['ctx_json']
    interactors = ctx.get('interactors', [])

    if not interactors:
        if verbose:
            print("[WARN]No interactors found, skipping cascade formatting")
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
        print("  ℹ No biological cascades found to format")
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

                            if verbose:
                                print(f"    [OK]Formatted: {func_ref['primary']} → {func_ref['function_name']}")
                                print(f"      Cascades: {len(new_cascades)}")

        except Exception as e:
            print(f"  [WARN]Cascade formatting failed for batch {batch_num + 1}: {e}")
            print(f"    Continuing with original cascades...")

    # Also update snapshot_json if it exists
    if 'snapshot_json' in json_data:
        json_data['snapshot_json']['interactors'] = interactors

    print(f"\n  [OK]Cascade formatting complete!")

    # Log step completion
    if step_logger:
        step_logger.log_terminal_output(f"\n  [OK]Cascade formatting complete!")
        step_metadata = {
            'functions_formatted': len(functions_to_format),
            'total_batches': total_batches
        }
        step_logger.log_step_complete(
            output_data=json_data,
            metadata=step_metadata,
            generate_summary=True
        )

    return json_data


def create_cascade_formatting_prompt(main_protein: str, functions: list) -> str:
    """
    Create a prompt for Gemini to format biological cascades.
    """
    prompt = f"""TASK: Format and validate biological cascades for scientific clarity and accuracy

You are a molecular biology expert reviewing biological cascade descriptions for a protein interaction database.

MAIN PROTEIN: {main_protein}

FORMATTING REQUIREMENTS:

1. **Logical Flow**: Each cascade must follow clear cause → effect progression
   - Start with molecular event (e.g., "ATXN3 binds VCP")
   - Include intermediate steps (e.g., "Displacement of Ufd1 cofactor")
   - End with cellular outcome (e.g., "ER stress")

2. **Arrow Notation**: Use consistent arrows (→) to separate steps
   - Each step should be clear and concise
   - Maximum 5-7 steps per cascade
   - If a cascade is too long (>7 steps), split into multiple related cascades

3. **Scientific Accuracy**: Ensure cascade steps are biologically valid
   - Include specific pathway names (e.g., "mTORC1 pathway", "UPR pathway")
   - Include affected proteins/complexes (e.g., "Beclin-1", "PtdIns3K complex")
   - Use proper molecular biology terminology
   - Verify cascade doesn't contradict cellular_process or specific_effects

4. **Clarity and Beauty**:
   - Each step should be readable and meaningful
   - Avoid overly technical jargon where simpler terms suffice
   - Ensure parallel structure across related cascades
   - Make it easy for researchers to understand the biological significance

5. **Consistency**: If function has multiple cascades, ensure they:
   - Don't contradict each other
   - Represent different pathways or contexts
   - Are all relevant to the function name

FUNCTIONS TO FORMAT:

"""

    for i, fn in enumerate(functions, 1):
        prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUNCTION {i}: {fn['primary']} → {fn['function_name']} (Effect: {fn['arrow']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cellular Process:
{fn['cellular_process'] or 'Not specified'}

Specific Effects:
{chr(10).join(f'• {effect}' for effect in fn['specific_effects']) if fn['specific_effects'] else '• None specified'}

Current Biological Cascades (TO BE FORMATTED):
{chr(10).join(f'{j}. {cascade}' for j, cascade in enumerate(fn['current_cascades'], 1))}
"""

    prompt += """

OUTPUT FORMAT:

Return a JSON object with this structure:

{
  "functions": [
    {
      "biological_consequence": [
        "Formatted cascade 1 with clear steps",
        "Formatted cascade 2 if multiple pathways exist"
      ]
    }
  ]
}

IMPORTANT:
- Return cascades in the SAME ORDER as input functions
- Each cascade should be a single string with arrows (→) separating steps
- Split overly long cascades into multiple entries
- Ensure all cascades are scientifically accurate
- Make cascades beautiful, clear, and meaningful
- If current cascade is already well-formatted, you may keep it (but improve if needed)

Return ONLY valid JSON, no markdown fences, no extra text.
"""

    return prompt


def main():
    """Main entry point for evidence validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate and enrich evidence in pipeline JSON output"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to the pipeline JSON file to validate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (default: <input>_validated.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of interactors to process per batch (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("[ERROR]GOOGLE_API_KEY not found. Add it to your .env file.")
    
    # Load input JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        sys.exit(f"[ERROR]Input file not found: {input_path}")
    
    print(f"\n{'='*80}")
    print("EVIDENCE VALIDATOR & CITATION ENRICHER")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    
    json_data = load_json_file(input_path)

    # Validate and enrich with timing
    start_time = time.time()
    try:
        validated_data = validate_and_enrich_evidence(
            json_data,
            api_key,
            verbose=args.verbose,
            batch_size=args.batch_size
        )
    except Exception as e:
        sys.exit(f"[ERROR]Validation failed: {e}")

    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_validated{input_path.suffix}"

    save_json_file(validated_data, output_path)

    print(f"\n{'='*80}")
    print("[OK]VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_min:.1f} minutes ({elapsed_time:.0f}s)")
    print(f"Output saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Review the validated JSON")
    print(f"  2. Generate visualization: python visualizer.py {output_path}")


if __name__ == "__main__":
    main()