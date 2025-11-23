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
MAX_THINKING_TOKENS_IDENTIFY = 2048  # Lower budget for simple identification
MAX_THINKING_TOKENS_GENERATE = 16384 # Higher budget for complex link generation with search
MAX_OUTPUT_TOKENS = 8192

def call_gemini(prompt: str, api_key: str, use_search: bool = False, thinking_budget: int = 2048) -> str:
    from google import genai as google_genai
    client = google_genai.Client(api_key=api_key)

    tools = []
    if use_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,
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
    """Extract JSON from text, handling markdown blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
    try:
        return json.loads(cleaned)
    except:
        # Try finding braces/brackets
        start_brace = cleaned.find('{')
        start_bracket = cleaned.find('[')

        start = -1
        if start_brace != -1 and start_bracket != -1:
            start = min(start_brace, start_bracket)
        elif start_brace != -1:
            start = start_brace
        elif start_bracket != -1:
            start = start_bracket

        if start != -1:
            # Find last matching closing
            end_brace = cleaned.rfind('}')
            end_bracket = cleaned.rfind(']')
            end = max(end_brace, end_bracket) + 1

            if end > start:
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

    print(f"\n{'='*60}")
    print(f"MEDIATOR LINKER: Analyzing {len(interactors)} interactors for {main_protein}")
    print(f"{'='*60}")

    # 1. Identify Candidates (Filter locally first)
    candidates_to_check = []

    for interactor in interactors:
        primary = interactor.get('primary')

        # Only process interactors marked as INDIRECT
        if interactor.get('interaction_type') != 'indirect':
            continue

        # Skip if already linked
        if interactor.get('upstream_interactor'):
            continue

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

        # Only check if there are potential mediators in the network (excluding self and main)
        potential_mediators = list(network_proteins - {primary, main_protein})
        if not potential_mediators:
            continue

        candidates_to_check.append({
            "primary": primary,
            "text": full_text[:2000], # Truncate to avoid token limits
            "potential_mediators": potential_mediators,
            "interactor_obj": interactor
        })

    if not candidates_to_check:
        print("No candidates for mediator linking found.")
        return payload, []

    print(f"Found {len(candidates_to_check)} candidates for mediator analysis.")

    # 2. Batch Identification (Parallelized)
    BATCH_SIZE = 10  # Increased batch size for speed

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Function to process a single batch
    def process_batch(batch_data):
        batch_items = []
        for item in batch_data:
            potential_list = ", ".join(item['potential_mediators'])
            batch_items.append(f"""
--- INTERACTOR: {item['primary']} ---
POTENTIAL MEDIATORS: {potential_list}
TEXT EVIDENCE:
{item['text']}
""")

        prompt = f"""
ANALYZE INDIRECT INTERACTIONS FOR MEDIATORS

MAIN PROTEIN: {main_protein}

You will be given a list of indirect interactors. For each, analyze the text evidence to see if one of the "POTENTIAL MEDIATORS" is explicitly named as the physical bridge/adaptor/complex partner for the interaction.

EXAMPLES:
Text: "CDC37 is essential for HSP90-mediated stabilization of TDP-43" (Potential: HSP90) -> MATCH: HSP90
Text: "TFEB regulates lysosomes via calcineurin" (Potential: PPP3CB) -> MATCH: PPP3CB (if PPP3CB is calcineurin subunit)
Text: "TFEB binds to DNA" (Potential: MTOR) -> NO MATCH

INPUTS:
{''.join(batch_items)}

TASK:
Return a JSON list of objects. Only include items where a mediator is found.
format: [{{ "target": "PRIMARY_SYMBOL", "mediator": "MEDIATOR_SYMBOL" }}, ...]
"""
        # Use ZERO thinking budget for simple identification to maximize speed
        response = call_gemini(prompt, api_key, use_search=False, thinking_budget=0)
        results = extract_json(response)

        if not isinstance(results, list):
            return []

        # Add source batch info to results for lookup
        processed_results = []
        for res in results:
            res['_source_batch'] = batch_data
            processed_results.append(res)

        return processed_results

    # Prepare batches
    batches = [candidates_to_check[i:i+BATCH_SIZE] for i in range(0, len(candidates_to_check), BATCH_SIZE)]

    # Run identification in parallel
    print(f"Processing {len(batches)} batches in parallel...")
    all_results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(future_to_batch):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                print(f"Batch identification generated an exception: {exc}")

    # 3. Process Matches and Generate Links (Sequential or Parallel? Parallel!)
    if not all_results:
        print("No mediators identified.")
        return payload, []

    print(f"Identified {len(all_results)} potential mediator links. Generating details...")

    def process_link_generation(res):
        target_symbol = res.get('target')
        mediator_symbol = res.get('mediator')
        source_batch = res.get('_source_batch')

        if not target_symbol or not mediator_symbol or not source_batch:
            return None

        # Find the corresponding interactor object
        candidate = next((c for c in source_batch if c['primary'] == target_symbol), None)

        if not candidate or mediator_symbol not in candidate['potential_mediators']:
            return None

        interactor = candidate['interactor_obj']

        print(f"  [LINK] Found mediator for {target_symbol}: {mediator_symbol}")

        # 1. Update Interactor in Payload (Thread-safe enough for dict updates usually, but let's be careful)
        # We update the shared dictionary directly
        interactor['upstream_interactor'] = mediator_symbol
        interactor['mediator_chain'] = [mediator_symbol]
        interactor['_linked_by_mediator_linker'] = True

        # 2. Generate Chain Link Interaction
        link_prompt = f"""
DESCRIBE INTERACTION: {mediator_symbol} <-> {target_symbol}
CONTEXT: They interact to regulate {main_protein} (specifically: {candidate['text'][:300]}...)

TASK:
Generate a structured interaction record for {mediator_symbol} and {target_symbol}.
Focus on THEIR direct interaction (e.g., "HSP90 binds CDC37 co-chaperone").
Use Google Search to ensure accuracy of their direct binding details.

OUTPUT JSON:
{{
    "primary": "{target_symbol}",
    "direction": "bidirectional",
    "arrow": "binds",
    "interaction_type": "direct",
    "confidence": 0.9,
    "functions": [
        {{
            "function": "interaction function (e.g. Co-chaperone complex formation)",
            "cellular_process": "Detailed mechanism of {mediator_symbol}-{target_symbol} binding",
            "arrow": "binds",
            "evidence": [
                {{ "paper_title": "Title", "relevant_quote": "Quote" }}
            ]
        }}
    ]
}}
"""
        # Use high thinking budget for generation as it requires search and detail
        link_response = call_gemini(link_prompt, api_key, use_search=True, thinking_budget=MAX_THINKING_TOKENS_GENERATE)
        link_data = extract_json(link_response)

        if link_data and isinstance(link_data, dict):
            # Ensure fields
            link_data['primary'] = target_symbol # Target relative to mediator

            return {
                'protein_a': mediator_symbol,
                'protein_b': target_symbol,
                'data': link_data
            }
        return None

    # Run generation in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_link = {executor.submit(process_link_generation, res): res for res in all_results}

        for future in as_completed(future_to_link):
            try:
                link_result = future.result()
                if link_result:
                    extra_interactions_to_save.append(link_result)
                    print(f"  [NEW] Generated link data for {link_result['protein_a']}-{link_result['protein_b']}")
            except Exception as exc:
                print(f"Link generation generated an exception: {exc}")

    return payload, extra_interactions_to_save

if __name__ == "__main__":
    print("Mediator Linker Module")
