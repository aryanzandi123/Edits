#!/usr/bin/env python3
"""
Chain Linker (New)
Analyzes indirect interactors to identify and link them to their mediator proteins.
If a mediator is found but not in the network, it RESEARCHES and ADDS it as a new direct interactor.
It also creates a specific interaction entry for the Mediator <-> Indirect link.
"""

import json
import time
from typing import Dict, List, Any, Tuple, Optional, Set
from google.genai import types
import re
from copy import deepcopy

# Config
MAX_THINKING_TOKENS = 8192
MAX_OUTPUT_TOKENS = 16384
BATCH_SIZE = 5

def call_gemini(prompt: str, api_key: str, use_search: bool = False, system_instruction: str = None) -> str:
    from google import genai as google_genai
    client = google_genai.Client(api_key=api_key)

    tools = []
    if use_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    config_dict = {
        "thinking_config": types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,
            include_thoughts=True,
        ),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.2,
    }

    if tools:
        config_dict["tools"] = tools

    if system_instruction:
        config_dict["system_instructions"] = types.Content(
             parts=[types.Part(text=system_instruction)]
        )

    config = types.GenerateContentConfig(**config_dict)

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
        print(f"[ChainLinker] LLM Error: {e}")
        return ""

def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]

    # Remove "json" prefix if present
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except:
        # Try finding braces/brackets
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

def identify_mediators(
    interactors: List[Dict],
    main_protein: str,
    existing_proteins: Set[str],
    api_key: str
) -> List[Dict]:
    """
    Identifies mediators for INDIRECT interactors.
    Returns a list of analysis results.
    """
    candidates_to_analyze = []

    for interactor in interactors:
        # Strict restriction: Only process INDIRECT interactors
        if interactor.get('interaction_type') != 'indirect':
            continue

        # Skip if already linked
        if interactor.get('upstream_interactor'):
            continue

        primary = interactor.get('primary')

        # Collect text evidence
        texts = []
        if interactor.get('support_summary'):
            texts.append(f"Summary: {interactor['support_summary']}")
        for f in interactor.get('functions', []):
            desc = []
            if f.get('function'): desc.append(f['function'])
            if f.get('cellular_process'): desc.append(f['cellular_process'])
            if f.get('mechanism'): desc.append(f['mechanism'])
            if desc:
                texts.append(f"Function: {'; '.join(desc)}")

        full_text = "\n".join(texts)
        if not full_text:
            continue

        candidates_to_analyze.append({
            'primary': primary,
            'text': full_text,
            'interactor_obj': interactor
        })

    if not candidates_to_analyze:
        return []

    results = []

    # Process in batches
    print(f"[ChainLinker] Analyzing {len(candidates_to_analyze)} indirect interactors for mediators...")

    for i in range(0, len(candidates_to_analyze), BATCH_SIZE):
        batch = candidates_to_analyze[i:i+BATCH_SIZE]

        batch_prompt = ""
        for idx, item in enumerate(batch):
            batch_prompt += f"""
--- ITEM {idx+1} ---
Target Indirect Interactor: {item['primary']}
Context (Evidence/Mechanism):
{item['text']}
"""

        prompt = f"""
You are analyzing biological interaction chains for the main protein {main_protein}.
The following proteins have been identified as INDIRECT interactors of {main_protein}.
Your job is to identify the "Mediator" protein that connects {main_protein} to these Indirect interactors.

The chain usually looks like: {main_protein} <-> [MEDIATOR] <-> [INDIRECT TARGET]

TASK:
1. Read the Context for each item carefully.
2. Identify if a specific Mediator protein is explicitly mentioned as the bridge or direct partner.
3. The Mediator MUST be a specific protein (e.g., "HSP90", "VCP", "CRM1").
4. Do NOT guess. If no mediator is clear, set "found": false.

OUTPUT JSON (list of objects):
[
    {{
        "item_index": 1,
        "target": "TARGET_SYMBOL",
        "found": true,
        "mediator_symbol": "SYMBOL" (Standardized Gene Symbol, e.g. HSP90AA1 or HSP90),
        "confidence": "high/medium/low",
        "reasoning": "quote or explanation"
    }},
    ...
]

ITEMS:
{batch_prompt}
"""
        response = call_gemini(prompt, api_key, use_search=False)
        batch_results = extract_json(response)

        if isinstance(batch_results, list):
            for res in batch_results:
                idx = res.get('item_index')
                if idx and 1 <= idx <= len(batch):
                    item = batch[idx-1]
                    if res.get('found') and res.get('mediator_symbol'):
                        mediator = res.get('mediator_symbol').upper()
                        # Self-check: Mediator cannot be the Main protein or the Target itself
                        if mediator != main_protein.upper() and mediator != item['primary'].upper():
                            results.append({
                                'target': item['primary'],
                                'interactor_obj': item['interactor_obj'],
                                'mediator': mediator,
                                'reasoning': res.get('reasoning')
                            })

    return results

def research_protein(protein_symbol: str, main_protein: str, api_key: str) -> Optional[Dict]:
    """
    Researches a new protein (Mediator) to add it as a full direct interactor.
    """
    print(f"[ChainLinker] Researching NEW Mediator: {protein_symbol}...")

    prompt = f"""
Comprehensive analysis of protein-protein interaction: {main_protein} and {protein_symbol}.

TASK:
1. Determine if {main_protein} and {protein_symbol} interact DIRECTLY (physically bind, form complex, or direct regulation).
2. If they interact, generate a detailed interaction record.

OUTPUT JSON:
{{
    "primary": "{protein_symbol}",
    "interaction_type": "direct",
    "direction": "bidirectional/main_to_primary/primary_to_main",
    "arrow": "binds/activates/inhibits",
    "intent": "binding/activation/inhibition",
    "confidence": 0.9,
    "support_summary": "Concise summary of how they interact.",
    "functions": [
        {{
            "function": "Specific biological function of this interaction",
            "cellular_process": "Detailed mechanism",
            "arrow": "binds/activates/inhibits",
            "pmids": ["12345678"],
            "evidence": [
                {{ "paper_title": "...", "relevant_quote": "..." }}
            ]
        }}
    ]
}}
"""
    response = call_gemini(prompt, api_key, use_search=True)
    data = extract_json(response)

    if data and isinstance(data, dict) and data.get('primary'):
        # Ensure minimal fields
        if 'functions' not in data: data['functions'] = []
        return data
    return None

def research_link(protein_a: str, protein_b: str, context: str, api_key: str) -> Optional[Dict]:
    """
    Researches the link between Mediator (A) and Indirect (B) in a specific context.
    """
    print(f"[ChainLinker] Researching Link: {protein_a} <-> {protein_b}...")

    prompt = f"""
Analyze the direct interaction between {protein_a} and {protein_b}.
Context of interest: {context}

TASK:
Describe the direct interaction (binding, regulation) between these two proteins.

OUTPUT JSON:
{{
    "primary": "{protein_b}",
    "interaction_type": "direct",
    "direction": "bidirectional",
    "arrow": "binds",
    "intent": "binding",
    "confidence": 0.9,
    "functions": [
        {{
            "function": "Interaction function (e.g. {protein_a}-{protein_b} complex)",
            "cellular_process": "Mechanism of their binding/interaction",
            "arrow": "binds",
            "pmids": [],
            "evidence": []
        }}
    ]
}}
"""
    # Note: "primary" in the output is B, relative to A.
    # But for the global list, we might need to adjust structure if we save it as a distinct entry.

    response = call_gemini(prompt, api_key, use_search=True)
    data = extract_json(response)

    if data and isinstance(data, dict):
        return data
    return None

def link_mediators_in_payload(
    payload: Dict[str, Any],
    api_key: str,
    verbose: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Main entry point.
    1. Identify mediators for indirect interactors.
    2. Ensure mediators exist in the network (add if missing).
    3. Create links between Mediator and Indirect.
    4. Update Indirect interactors with upstream info.
    """
    if 'ctx_json' not in payload:
        return payload, []

    ctx = payload['ctx_json']
    interactors = ctx.get('interactors', [])
    main_protein = ctx.get('main', 'UNKNOWN')

    # Map of existing proteins (for quick lookup)
    existing_interactors_map = {i.get('primary'): i for i in interactors if i.get('primary')}
    existing_proteins = set(existing_interactors_map.keys())
    existing_proteins.add(main_protein)

    # 1. Identify Candidates
    analysis_results = identify_mediators(interactors, main_protein, existing_proteins, api_key)

    if not analysis_results:
        return payload, []

    extra_interactions_to_save = [] # List of {protein_a, protein_b, data}

    for res in analysis_results:
        target_indirect = res['target'] # The indirect protein
        mediator = res['mediator']      # The mediator protein
        interactor_obj = res['interactor_obj']

        # 2. Check if Mediator exists
        if mediator not in existing_proteins:
            # RESEARCH NEW MEDIATOR
            new_mediator_data = research_protein(mediator, main_protein, api_key)
            if new_mediator_data:
                # Add to main list
                interactors.append(new_mediator_data)
                existing_proteins.add(mediator)
                existing_interactors_map[mediator] = new_mediator_data
                print(f"[ChainLinker] Added NEW direct interactor (Mediator): {mediator}")
            else:
                print(f"[ChainLinker] Failed to research mediator {mediator}. Skipping link.")
                continue

        # 3. Research Link (Mediator <-> Indirect)
        # We need a dedicated interaction entry for the graph.
        # This entry represents the edge Mediator -> Indirect.

        # Determine context string
        context_str = interactor_obj.get('support_summary', '')

        link_data = research_link(mediator, target_indirect, context_str, api_key)

        if link_data:
            # Create edge object
            edge_obj = {
                'protein_a': mediator,
                'protein_b': target_indirect,
                'data': link_data
            }

            # Add to extra_interactions_to_save for the database
            extra_interactions_to_save.append(edge_obj)

            # 4. Link in Graph (Visualizer Support)
            # We inject these extra edges into a temporary stash in payload['ctx_json']
            # so runner.py can pick them up during snapshot creation.
            if 'extra_edges' not in ctx:
                ctx['extra_edges'] = []
            ctx['extra_edges'].append(edge_obj)

        # 5. Update Indirect Interactor
        interactor_obj['upstream_interactor'] = mediator
        interactor_obj['mediator_chain'] = [mediator]
        interactor_obj['_linked_by_chain_linker'] = True
        print(f"[ChainLinker] Linked {target_indirect} via mediator {mediator}")

    return payload, extra_interactions_to_save

if __name__ == "__main__":
    print("Chain Linker Module (New)")
