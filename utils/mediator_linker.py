#!/usr/bin/env python3
"""
Mediator Linker
Analyzes indirect interactors to identify and link them to their mediator proteins
if the mediator is present in the interaction network.
"""

import json
import time
from typing import Dict, List, Any, Tuple, Optional
from google.genai import types
import re

# Config
MAX_THINKING_TOKENS = 16384
MAX_OUTPUT_TOKENS = 8192
BATCH_SIZE = 5

def call_gemini(prompt: str, api_key: str, use_search: bool = False) -> str:
    from google import genai as google_genai
    client = google_genai.Client(api_key=api_key)

    tools = []
    if use_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,
            include_thoughts=True,
        ),
        tools=tools,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            return ''.join(part.text for part in parts if hasattr(part, 'text')).strip()
        return ""
    except Exception as e:
        print(f"[MediatorLinker] LLM Error: {e}")
        return ""

def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
    try:
        return json.loads(cleaned)
    except:
        # Try finding braces/brackets
        # Check if it looks like a list
        start_bracket = cleaned.find('[')
        start_brace = cleaned.find('{')

        start = -1
        end = -1

        if start_bracket >= 0 and (start_brace == -1 or start_bracket < start_brace):
             start = start_bracket
             end = cleaned.rfind(']') + 1
        elif start_brace >= 0:
             start = start_brace
             end = cleaned.rfind('}') + 1

        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except:
                pass
        return {}

def link_mediators_in_payload(
    payload: Dict[str, Any],
    api_key: str,
    verbose: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Scans interactors for mentioned mediators that exist in the network.
    Updates the indirect interactor with upstream_interactor.
    Returns updated payload and a list of NEW interaction objects (Mediator-Target) to save.

    Returns:
        (updated_payload, list_of_extra_interactions)
        extra_interactions = [{'protein_a': 'HSP90', 'protein_b': 'CDC37', 'data': {...}}]
    """
    if 'ctx_json' not in payload:
        return payload, []

    ctx = payload['ctx_json']
    interactors = ctx.get('interactors', [])
    main_protein = ctx.get('main', 'UNKNOWN')

    # Map of all protein symbols in the network
    network_proteins = {i.get('primary') for i in interactors if i.get('primary')}
    network_proteins.add(main_protein)

    extra_interactions_to_save = []

    # Filter candidates for analysis
    candidates_to_analyze = []

    # 1. Collect all valid candidates first
    for interactor in interactors:
        # Only process interactors marked as INDIRECT
        if interactor.get('interaction_type') != 'indirect':
            continue

        # Skip if already linked
        if interactor.get('upstream_interactor'):
            continue

        primary = interactor.get('primary')

        # Collect text evidence
        texts = []
        if interactor.get('support_summary'):
            texts.append(interactor['support_summary'])
        for f in interactor.get('functions', []):
            if f.get('cellular_process'): texts.append(f['cellular_process'])
            if f.get('mechanism'): texts.append(f['mechanism'])

        full_text = "\n".join(texts)
        if not full_text:
            continue

        # Check if there are any potential mediators (other than self and main)
        candidates = list(network_proteins - {primary, main_protein})
        if not candidates:
            continue

        candidates_to_analyze.append({
            'interactor': interactor,
            'primary': primary,
            'text': full_text,
            'candidates': candidates
        })

    print(f"\n{'='*60}")
    print(f"MEDIATOR LINKER: Analyzing {len(interactors)} interactors for {main_protein}")
    print(f"Candidates for linking: {len(candidates_to_analyze)}")
    print(f"{'='*60}")

    if not candidates_to_analyze:
        return payload, []

    # 2. Process in batches
    for i in range(0, len(candidates_to_analyze), BATCH_SIZE):
        batch = candidates_to_analyze[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(candidates_to_analyze)-1)//BATCH_SIZE + 1} ({len(batch)} items)")

        # Use candidates from the first item (approximation, but usually network is same)
        # Actually, candidates might vary if we exclude 'primary', but usually the set of 'other proteins' is large.
        # Let's use the union of all candidates or just the first one's list if simple.
        # Better: Pass the full list of potential mediators (network_proteins - main_protein) and ask LLM to pick, excluding the target itself.
        potential_mediators = list(network_proteins - {main_protein})

        batch_items_str = ""
        for idx, item in enumerate(batch):
            batch_items_str += f"""
ITEM {idx+1}:
Target: {item['primary']}
Text:
{item['text']}
---
"""

        prompt = f"""
ANALYZE INTERACTION MECHANISMS: {main_protein} -> Targets

POTENTIAL MEDIATORS (Proteins present in current network):
{", ".join(potential_mediators)}

ITEMS TO ANALYZE:
{batch_items_str}

TASK:
For each item, determine if the text explicitly states that one of the "POTENTIAL MEDIATORS" acts as a physical mediator/bridge/complex partner for the interaction between {main_protein} and the Target.
Note: The mediator MUST be different from the Target.

OUTPUT JSON (list of objects):
[
    {{
        "item_index": 1,
        "target": "TARGET_SYMBOL",
        "mediator_found": true/false,
        "mediator_symbol": "SYMBOL" or null,
        "reasoning": "Explanation"
    }},
    ...
]
"""
        response = call_gemini(prompt, api_key, use_search=False)
        results = extract_json(response)

        if not isinstance(results, list):
            # Fallback for single object response
            if isinstance(results, dict):
                results = [results]
            else:
                print(f"[MediatorLinker] Invalid JSON response format: {type(results)}")
                continue

        # Map results back to items
        for res in results:
            if not isinstance(res, dict):
                continue

            idx = res.get('item_index')
            if idx is None:
                # Try to match by target name
                target = res.get('target')
                for j, item in enumerate(batch):
                    if item['primary'] == target:
                        idx = j + 1
                        break

            if idx is None or idx < 1 or idx > len(batch):
                continue

            item = batch[idx-1]
            interactor = item['interactor']
            primary = item['primary']
            full_text = item['text']

            if res.get('mediator_found'):
                mediator = res.get('mediator_symbol')

                # Validate mediator
                if mediator in network_proteins and mediator != primary:
                    print(f"  [LINK] Found mediator for {primary}: {mediator}")

                    # 1. Update Interactor in Payload
                    interactor['interaction_type'] = 'indirect'
                    interactor['upstream_interactor'] = mediator
                    interactor['mediator_chain'] = [mediator]
                    interactor['_linked_by_mediator_linker'] = True

                    # 2. Generate Chain Link Interaction (Mediator <-> Target)
                    # We need to describe the interaction between Mediator and Target in this context
                    link_prompt = f"""
DESCRIBE INTERACTION: {mediator} <-> {primary}
CONTEXT: They interact to regulate {main_protein} (specifically: {full_text[:300]}...)

TASK:
Generate a structured interaction record for {mediator} and {primary}.
Focus on THEIR direct interaction (e.g., "HSP90 binds CDC37 co-chaperone").
Use Google Search to ensure accuracy of their direct binding details.

OUTPUT JSON:
{{
    "primary": "{primary}",
    "direction": "bidirectional",
    "arrow": "binds",
    "interaction_type": "direct",
    "confidence": 0.9,
    "functions": [
        {{
            "function": "interaction function (e.g. Co-chaperone complex formation)",
            "cellular_process": "Detailed mechanism of {mediator}-{primary} binding",
            "arrow": "binds",
            "evidence": [
                {{ "paper_title": "Title", "relevant_quote": "Quote" }}
            ]
        }}
    ]
}}
"""
                    link_response = call_gemini(link_prompt, api_key, use_search=True)
                    link_data = extract_json(link_response)

                    if link_data and isinstance(link_data, dict):
                        # Ensure fields
                        link_data['primary'] = primary # Target relative to mediator

                        # Add to extra interactions
                        extra_interactions_to_save.append({
                            'protein_a': mediator,
                            'protein_b': primary,
                            'data': link_data
                        })
                        print(f"  [NEW] Generated link data for {mediator}-{primary}")

    return payload, extra_interactions_to_save

if __name__ == "__main__":
    print("Mediator Linker Module")
