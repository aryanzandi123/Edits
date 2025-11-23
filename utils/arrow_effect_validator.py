"""
Arrow & Effect Validator
========================

Validates arrow notation, interaction directions, and effects for protein-protein interactions
using Gemini 2.5 Pro with thinking mode and Google Search.

Ensures:
- Correct arrow types (activates/inhibits/binds/regulates)
- Correct interaction directions (main_to_primary/primary_to_main/bidirectional)
- Correct interaction_effect alignment with arrow
- No double-negative issues (e.g., "inhibits Apoptosis Inhibition" → "activates Apoptosis Inhibition")
- Logical biological consequence chains

Processes 3-4 interactions in parallel for efficiency.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check if Gemini is available
try:
    from google import genai as google_genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Arrow validation will be skipped.")


# Constants
MAX_THINKING_TOKENS = 32768  # Generous thinking budget for complex validation
MAX_OUTPUT_TOKENS = 8192     # Sufficient for corrections JSON
TEMPERATURE = 0.2            # Deterministic corrections
TOP_P = 0.90
MAX_WORKERS = 4              # Parallel validation workers

# Valid values reference
VALID_DIRECTIONS = ["main_to_primary", "primary_to_main", "bidirectional"]
VALID_ARROWS = ["activates", "inhibits", "binds", "regulates", "complex"]
VALID_INTERACTION_TYPES = ["direct", "indirect"]


def validate_arrows_and_effects(
    payload: Dict[str, Any],
    api_key: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main entry point: validates all interactions in the payload.

    Args:
        payload: Full pipeline payload with snapshot_json and ctx_json
        api_key: Google AI API key
        verbose: Enable detailed logging

    Returns:
        Updated payload with corrected arrows/directions/effects
    """
    if not GEMINI_AVAILABLE:
        if verbose:
            print("[SKIP] Arrow validation disabled (Gemini not available)")
        return payload

    if not api_key:
        if verbose:
            print("[SKIP] Arrow validation disabled (no API key)")
        return payload

    # Extract interactors from snapshot_json
    snapshot = payload.get("snapshot_json", payload)
    main_protein = snapshot.get("main", "UNKNOWN")
    interactors = snapshot.get("interactors", [])

    if not interactors:
        if verbose:
            print("[SKIP] No interactors to validate")
        return payload

    if verbose:
        print(f"\n{'='*60}")
        print(f"ARROW VALIDATION: {main_protein} ({len(interactors)} interactions)")
        print(f"{'='*60}")

    # Determine worker count (max 4 concurrent workers)
    worker_count = max(1, min(MAX_WORKERS, len(interactors)))

    # Validate interactions in parallel
    corrected_interactors = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                validate_single_interaction,
                interactor,
                main_protein,
                api_key,
                verbose
            ): idx
            for idx, interactor in enumerate(interactors)
        }

        # Collect results as they complete
        results = [None] * len(interactors)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                corrected = future.result()
                results[idx] = corrected
                if verbose:
                    partner = corrected.get("primary", "UNKNOWN")
                    print(f"  ✓ Validated {main_protein} ↔ {partner}")
            except Exception as exc:
                partner = interactors[idx].get("primary", "UNKNOWN")
                print(f"  ✗ Error validating {main_protein} ↔ {partner}: {exc}")
                results[idx] = interactors[idx]  # Keep original on error

        corrected_interactors = [r for r in results if r is not None]

    # Update payload
    snapshot["interactors"] = corrected_interactors
    payload["snapshot_json"] = snapshot

    if verbose:
        print(f"{'='*60}")
        print(f"ARROW VALIDATION COMPLETE: {len(corrected_interactors)}/{len(interactors)} validated")
        print(f"{'='*60}\n")

    return payload


def validate_single_interaction(
    interactor: Dict[str, Any],
    main_protein: str,
    api_key: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validates a single protein-protein interaction using Gemini.

    Args:
        interactor: Interaction data for one partner protein
        main_protein: Query protein symbol
        api_key: Google AI API key
        verbose: Enable detailed logging

    Returns:
        Corrected interactor data
    """
    partner = interactor.get("primary", "UNKNOWN")

    try:
        # Build validation prompt
        prompt = build_validation_prompt(interactor, main_protein)

        # Call Gemini with thinking mode + Google Search
        client = google_genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=MAX_THINKING_TOKENS,
                include_thoughts=True,
            ),
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )

        # Parse corrections from response
        corrections = parse_gemini_response(response)

        # Apply corrections to interactor
        if corrections:
            interactor = apply_corrections(interactor, corrections, main_protein, verbose)
            if verbose:
                print(f"    → Applied {len(corrections)} correction(s) to {partner}")

        return interactor

    except Exception as e:
        print(f"[ERROR] Failed to validate {main_protein} ↔ {partner}: {e}")
        return interactor  # Return original on error


def build_validation_prompt(interactor: Dict[str, Any], main_protein: str) -> str:
    """
    Builds a detailed validation prompt for Gemini.

    Args:
        interactor: Interaction data
        main_protein: Query protein symbol

    Returns:
        Formatted prompt string
    """
    partner = interactor.get("primary", "UNKNOWN")
    direction = interactor.get("direction", "unknown")
    arrow = interactor.get("arrow", "unknown")
    interaction_type = interactor.get("interaction_type", "direct")

    # Extract chain data for indirect interactions
    upstream_interactor = interactor.get("upstream_interactor")
    mediator_chain = interactor.get("mediator_chain", [])
    depth = interactor.get("depth", 1)
    is_indirect = (interaction_type == "indirect" and mediator_chain)

    # Extract function data
    functions = interactor.get("functions", [])
    function_summary = []
    missing_function_arrows = []  # Track functions with missing arrows

    for idx, func in enumerate(functions):
        func_name = func.get("function", "Unknown")
        func_arrow = func.get("arrow", "")
        func_effect = func.get("interaction_effect", "unknown")
        func_direction = func.get("interaction_direction", "unknown")
        consequences = func.get("biological_consequence", [])

        # Detect missing or empty function arrows
        if not func_arrow or func_arrow == "unknown":
            missing_function_arrows.append({
                "index": idx,
                "function": func_name,
                "current_arrow": func_arrow,
                "interaction_arrow": arrow
            })
            func_arrow = f"MISSING (falls back to interaction '{arrow}')"

        function_summary.append({
            "name": func_name,
            "arrow": func_arrow,
            "interaction_effect": func_effect,
            "interaction_direction": func_direction,
            "consequences": consequences[:2]  # Limit to 2 for brevity
        })

    # Extract evidence (paper titles only)
    evidence = interactor.get("evidence", [])
    paper_titles = [ev.get("paper_title", "Unknown") for ev in evidence[:3]]  # Limit to 3

    # Build chain context section for indirect interactions
    chain_context = ""
    if is_indirect:
        chain_str = " → ".join([main_protein] + mediator_chain + [partner])
        chain_context = f"""

**⚠️ INDIRECT INTERACTION (via mediator chain):**

This is NOT a direct interaction between {main_protein} and {partner}.
The effect occurs through intermediary protein(s).

**Chain Structure:**
- Full pathway: {chain_str}
- Upstream interactor: {upstream_interactor} (last protein before {partner})
- Chain depth: {depth}
- Mediator chain: {' → '.join(mediator_chain)}

**CRITICAL VALIDATION RULES FOR INDIRECT INTERACTIONS:**

1. **Net Arrow Logic:**
   - The 'arrow' field ({arrow}) represents the NET EFFECT of {main_protein} on {partner}
   - This may DIFFER from the direct effect of {upstream_interactor} on {partner}
   - Example: If {main_protein} activates an inhibitor ({upstream_interactor}),
     the net effect on {partner} is INHIBITION (activating inhibitor = net inhibition)
   - Count inhibitory steps: Even number = net activation, Odd number = net inhibition

2. **Function Arrows (CRITICAL - DUAL ARROWS FOR INDIRECT):**
   For indirect interactions, you MUST provide TWO arrows per function:

   a) **NET ARROW** (arrow field):
      - Describes {main_protein}'s effect on the FUNCTION through the full chain
      - Consider the FULL CHAIN context when validating
      - Example: If {main_protein} activates an inhibitor that inhibits a function,
        net arrow is "inhibits" (activating inhibitor = net inhibition)

   b) **DIRECT ARROW** (direct_arrow field - NEW):
      - Describes {upstream_interactor}'s DIRECT effect on the FUNCTION
      - Independent of the chain - just the immediate mediator's effect
      - Example: If {upstream_interactor} directly inhibits {partner}'s apoptosis,
        direct_arrow is "inhibits"

   **Both arrows must be included in your response for each function!**

3. **Chain Consistency Checks:**
   - Verify that {upstream_interactor} is the correct last step in the chain
   - Check if mediator chain makes biological sense
   - Validate that net arrow matches expected chain logic
   - Validate that direct arrow matches {upstream_interactor} → {partner} relationship

4. **Google Search Strategy:**
   - Search for direct link: "{upstream_interactor} {partner} interaction"
   - Search for net effect: "{main_protein} {partner} pathway"
   - Search for mechanism: "{' '.join(mediator_chain)} {partner} regulation"
   - Verify both effects separately

**Important:**
- Net arrow should reflect the COMBINED effect through ALL steps in the chain
- Direct arrow should reflect ONLY the {upstream_interactor} → {partner} relationship
"""

    # Build prompt
    prompt = f"""You are a molecular biology expert validating protein interaction notation.

**TASK:** Check the following interaction for correctness and logical consistency.

**PROTEINS:**
- Main Protein: {main_protein}
- Partner Protein: {partner}

**CURRENT ANNOTATION:**
- Interaction Direction: {direction}
- Interaction Arrow: {arrow}
- Interaction Type: {interaction_type}
{chain_context}
**FUNCTIONS (with arrows and effects):**
{json.dumps(function_summary, indent=2)}

**SUPPORTING EVIDENCE (paper titles):**
{json.dumps(paper_titles, indent=2)}

---

**VALIDATION REQUIREMENTS:**

1. **Direction Accuracy:**
   - Valid values: "main_to_primary" | "primary_to_main" | "bidirectional"
   - Check: Does the direction match the biological mechanism described in functions?
   - Example: If {main_protein} phosphorylates {partner}, direction should be "main_to_primary"

2. **Interaction Arrow (Protein-Level Effect):**
   - Valid values: "activates" | "inhibits" | "binds" | "regulates" | "complex"
   - Check: Does the arrow match the predominant effect across functions?
   - Example: If most functions show inhibition, arrow should be "inhibits"
   - This describes the effect on the PARTNER PROTEIN

3. **Function Arrows (CRITICAL - Can Differ from Interaction Arrow):**
   - EVERY function MUST have its own "arrow" field
   - Function arrow describes effect ON THE FUNCTION, not on the protein
   - Function arrows can and should differ from interaction arrows

   **Examples:**
   - Interaction: {main_protein} → {partner} (binds)
   - Function: "Apoptosis Promotion" with arrow: "inhibits"
     → Correct: {main_protein} binds {partner} and thereby inhibits apoptosis promotion

   - Interaction: VCP → ATXN3 (binds)
   - Function: "Promotion of Pathogenic ATXN3 Aggregation" with arrow: "inhibits"
     → Correct: VCP binds ATXN3 but inhibits/prevents the promotion of aggregation

   **Missing Function Arrows:**
   {json.dumps(missing_function_arrows, indent=2) if missing_function_arrows else "None - all functions have arrows"}

4. **Double-Negative Detection (CRITICAL):**
   - Check each function name + arrow combination
   - Examples of double negatives to fix:
     * Function: "Apoptosis Inhibition" + Arrow: "inhibits" → FIX: Arrow should be "activates"
     * Function: "Cell Death Suppression" + Arrow: "inhibits" → FIX: Arrow should be "activates"
     * Function: "Degradation Prevention" + Arrow: "inhibits" → FIX: Arrow should be "activates"
     * Function: "Proliferation" + Arrow: "inhibits" → OK (no double negative)
   - Rule: If function name contains negative terms (inhibition, suppression, repression, prevention, etc.)
     and arrow is "inhibits", change arrow to "activates" (inhibiting an inhibitor = activation)

5. **Function-Level Consistency:**
   - Check: Does `interaction_effect` match `arrow` for each function?
   - Check: Does `interaction_direction` align with the main interaction direction?
   - Check: Are `biological_consequence` chains logically sound (A → B → C)?

6. **Interaction Type:**
   - Valid values: "direct" | "indirect"
   - Check: Does this match the evidence (direct binding vs pathway-mediated)?

---

**INSTRUCTIONS:**

1. **PRIORITY:** Populate missing function arrows (marked as "MISSING" above)
2. For each missing arrow, determine the correct effect based on:
   - Function name semantics (e.g., "Promotion" vs "Inhibition")
   - Mechanism description
   - Biological context from evidence
3. Use Google Search to verify mechanisms if uncertain about biological accuracy
4. Search for: "{main_protein} {partner} interaction mechanism" and "{main_protein} {partner} [function name]"
5. Focus on recent papers (2015+) and review articles
6. Return ONLY corrections in JSON format (omit unchanged fields)
7. If no corrections needed, return empty JSON: {{}}

**OUTPUT FORMAT:**

```json
{{
  "interaction_level": {{
    "direction": "corrected_value",  // Only if changed
    "arrow": "corrected_value",      // Only if changed
    "interaction_effect": "corrected_value",  // Only if changed (should match arrow)
    "interaction_type": "corrected_value"  // Only if changed
  }},
  "functions": [
    {{
      "function": "Function Name Here",  // For identification
      "corrections": {{
        "arrow": "corrected_value",  // Net arrow (REQUIRED for all)
        "direct_arrow": "corrected_value",  // ONLY for indirect interactions - mediator's direct effect
        "interaction_effect": "corrected_value",
        "interaction_direction": "corrected_value"
      }},
      "reasoning": "Brief explanation of why this was changed"
    }}
  ],
  "validation_summary": "Overall assessment (1-2 sentences)"
}}
```

**IMPORTANT:**
- Only include fields that need correction
- Preserve biological accuracy over notation consistency
- If uncertain, err on the side of "regulates" for complex interactions
- Double-check for double negatives (most common error)

Begin validation:"""

    return prompt


def parse_gemini_response(response) -> Optional[Dict[str, Any]]:
    """
    Extracts corrections JSON from Gemini response.

    Args:
        response: Gemini API response object

    Returns:
        Corrections dict or None if no corrections
    """
    try:
        # Get response text
        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            text = response.candidates[0].content.parts[0].text
        else:
            return None

        # Extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None

        # Parse JSON
        corrections = json.loads(json_str)

        # Return None if empty corrections
        if not corrections or corrections == {}:
            return None

        return corrections

    except Exception as e:
        print(f"[WARNING] Failed to parse Gemini response: {e}")
        return None


def apply_corrections(
    interactor: Dict[str, Any],
    corrections: Dict[str, Any],
    main_protein: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Applies corrections to interactor data.

    Args:
        interactor: Original interactor data
        corrections: Corrections dict from Gemini
        main_protein: Query protein symbol
        verbose: Enable detailed logging

    Returns:
        Updated interactor data
    """
    partner = interactor.get("primary", "UNKNOWN")

    # Apply interaction-level corrections
    interaction_corrections = corrections.get("interaction_level", {})
    for field, new_value in interaction_corrections.items():
        old_value = interactor.get(field, None)
        if old_value != new_value:
            interactor[field] = new_value
            if verbose:
                print(f"      ✎ {field}: {old_value} → {new_value}")

    # Auto-generate interaction_effect from arrow if not set
    arrow = interactor.get("arrow", "")
    if not interactor.get("interaction_effect") and arrow:
        # Map arrow to effect: activates → activation, inhibits → inhibition, binds → binding
        effect_map = {
            "activates": "activation",
            "inhibits": "inhibition",
            "binds": "binding",
            "regulates": "regulation",
            "complex": "complex formation"
        }
        interaction_effect = effect_map.get(arrow, arrow)
        interactor["interaction_effect"] = interaction_effect
        if verbose:
            print(f"      ✎ interaction_effect: (auto-generated) → {interaction_effect}")

    # Apply function-level corrections
    function_corrections = corrections.get("functions", [])
    functions = interactor.get("functions", [])

    for func_correction in function_corrections:
        func_name = func_correction.get("function", "")
        func_changes = func_correction.get("corrections", {})
        reasoning = func_correction.get("reasoning", "")

        # Find matching function
        for func in functions:
            if func.get("function", "") == func_name:
                # Apply corrections
                for field, new_value in func_changes.items():
                    old_value = func.get(field, None)
                    if old_value != new_value:
                        func[field] = new_value
                        if verbose:
                            print(f"      ✎ {func_name}.{field}: {old_value} → {new_value}")
                            if reasoning and field in ["arrow", "direct_arrow"]:
                                print(f"        → {reasoning}")
                break

    interactor["functions"] = functions

    # Auto-generate function_effect for each function based on arrow
    effect_map = {
        "activates": "activation",
        "inhibits": "inhibition",
        "binds": "binding",
        "regulates": "regulation",
        "complex": "complex formation"
    }

    for func in functions:
        func_arrow = func.get("arrow", "")
        if func_arrow and not func.get("function_effect"):
            func["function_effect"] = effect_map.get(func_arrow, func_arrow)

    # Add dual arrow context for indirect interactions
    interaction_type = interactor.get("interaction_type", "direct")
    if interaction_type == "indirect":
        # For indirect interactions, add arrow_context to each function
        upstream_interactor = interactor.get("upstream_interactor")
        mediator_chain = interactor.get("mediator_chain", [])
        main_protein_symbol = main_protein  # query protein from outer scope

        for func in functions:
            # The 'arrow' field represents the NET effect (query → target function)
            net_arrow = func.get("arrow", "regulates")

            # The 'direct_arrow' field (if set by Gemini) represents the DIRECT effect
            # (mediator → target function). If not set, default to net_arrow.
            # WARN if missing - Gemini should provide both for indirect interactions!
            if "direct_arrow" not in func:
                if verbose:
                    func_name = func.get("function", "Unknown")
                    print(f"      ⚠ INDIRECT: {func_name} missing direct_arrow, defaulting to net_arrow ({net_arrow})")
                direct_arrow = net_arrow
            else:
                direct_arrow = func["direct_arrow"]

            # Add arrow_context with both perspectives
            func["arrow_context"] = {
                "direct_from": upstream_interactor,  # Last protein in chain (mediator)
                "direct_arrow": direct_arrow,  # Mediator's direct effect
                "net_from": main_protein_symbol,  # Query protein
                "net_arrow": net_arrow,  # Query's net effect through chain
                "mediator_chain": mediator_chain,
                "is_indirect": True
            }

            # Store both arrows at function level for easy access
            func["net_arrow"] = net_arrow
            func["direct_arrow"] = direct_arrow

            # Keep 'arrow' field as net_arrow for backward compatibility
            func["arrow"] = net_arrow

            # Generate separate effect labels for both perspectives
            effect_map = {
                "activates": "activation",
                "inhibits": "inhibition",
                "binds": "binding",
                "regulates": "regulation",
                "complex": "complex formation"
            }
            func["net_effect"] = effect_map.get(net_arrow, net_arrow)
            func["direct_effect"] = effect_map.get(direct_arrow, direct_arrow)

    # Add validation metadata
    interactor["_validation_metadata"] = {
        "validated": True,
        "validator": "arrow_effect_validator",
        "corrections_applied": len(interaction_corrections) + len(function_corrections)
    }

    # Add top-level arrow validation flag (makes it easy for subsequent stages to check)
    interactor["_arrow_validated"] = True

    return interactor


# Standalone test function
def test_validator():
    """Test the validator on a sample interaction."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        return

    # Sample interaction with double-negative issue
    sample_interactor = {
        "primary": "VCP",
        "direction": "main_to_primary",
        "arrow": "inhibits",
        "interaction_type": "direct",
        "functions": [
            {
                "function": "ER-Associated Degradation (ERAD) Inhibition",
                "arrow": "inhibits",  # DOUBLE NEGATIVE: should be "activates"
                "interaction_effect": "inhibits",
                "interaction_direction": "main_to_primary",
                "biological_consequence": [
                    "ATXN3 inhibits VCP → ERAD inhibition increases → Protein accumulation"
                ]
            }
        ],
        "evidence": [
            {
                "paper_title": "Ataxin-3 binds VCP/p97 and regulates retrotranslocation of ERAD substrates",
                "year": 2006
            }
        ]
    }

    print("\nTesting arrow validator...")
    print(f"Original arrow: {sample_interactor['arrow']}")
    print(f"Original function arrow: {sample_interactor['functions'][0]['arrow']}")

    corrected = validate_single_interaction(
        sample_interactor,
        "ATXN3",
        api_key,
        verbose=True
    )

    print(f"\nCorrected arrow: {corrected['arrow']}")
    print(f"Corrected function arrow: {corrected['functions'][0]['arrow']}")
    print("\nTest complete!")


if __name__ == "__main__":
    test_validator()
