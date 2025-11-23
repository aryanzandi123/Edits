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
BATCH_SIZE = 5

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

def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
    try:
        return json.loads(cleaned)
    except:
        # Try finding braces
        start = cleaned.find('{')
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

    print(f"\n{'='*60}")
    print(f"MEDIATOR LINKER: Analyzing {len(interactors)} interactors for {main_protein}")
    print(f"{'='*60}")

    for interactor in interactors:
        primary = interactor.get('primary')

        # Skip if already linked
        if interactor.get('upstream_interactor'):
            continue

        # Skip if direct with no hint of being indirect?
        # Actually, sometimes things are marked direct but are physically indirect.
        # But let's focus on items marked 'indirect' OR items with text suggesting a complex.
        # User said "many indirect interactors... were NOT connected".

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

        # Prompt LLM to find mediator
        candidates = list(network_proteins - {primary, main_protein})
        if not candidates:
            continue

        prompt = f"""
ANALYZE INTERACTION MECHANISM: {main_protein} -> {primary}

TEXT EVIDENCE:
{full_text}

POTENTIAL MEDIATORS (Proteins present in current network):
{", ".join(candidates)}

TASK:
1. Does the text explicitly state that one of the "POTENTIAL MEDIATORS" acts as a physical mediator/bridge/complex partner for this interaction?
2. Example: "CDC37 is essential for HSP90-mediated stabilization of TDP-43". Here HSP90 is the mediator for CDC37's effect on TDP-43.
3. If yes, extract the mediator symbol.

OUTPUT JSON:
{{
    "mediator_found": true/false,
    "mediator_symbol": "SYMBOL",
    "reasoning": "Explanation"
}}
"""
        response = call_gemini(prompt, api_key, use_search=False)
        result = extract_json(response)

        if result.get('mediator_found') and result.get('mediator_symbol') in network_proteins:
            mediator = result['mediator_symbol']
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

            if link_data:
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

def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
    try:
        return json.loads(cleaned)
    except:
        # Try finding braces
        start = cleaned.find('{')
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

        # Prompt LLM to find mediator
        candidates = list(network_proteins - {primary, main_protein})
        if not candidates:
            continue

        prompt = f"""
ANALYZE INTERACTION MECHANISM: {main_protein} -> {primary}

TEXT EVIDENCE:
{full_text}

POTENTIAL MEDIATORS (Proteins present in current network):
{", ".join(candidates)}

TASK:
1. Does the text explicitly state that one of the "POTENTIAL MEDIATORS" acts as a physical mediator/bridge/complex partner for this interaction?
2. Example: "CDC37 is essential for HSP90-mediated stabilization of TDP-43". Here HSP90 is the mediator for CDC37's effect on TDP-43.
3. If yes, extract the mediator symbol.

OUTPUT JSON:
{{
    "mediator_found": true/false,
    "mediator_symbol": "SYMBOL",
    "reasoning": "Explanation"
}}
"""
        response = call_gemini(prompt, api_key, use_search=False)
        result = extract_json(response)

        if result.get('mediator_found') and result.get('mediator_symbol') in network_proteins:
            mediator = result['mediator_symbol']
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
            link_response = call_gemini(link_prompt, api_key, use_search=True)
            link_data = extract_json(link_response)

            if link_data:
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
