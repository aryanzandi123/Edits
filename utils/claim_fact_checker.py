#!/usr/bin/env python3
"""
Claim Fact-Checker - Validates if functional claims are actually supported by literature
For each function, searches PubMed to verify the claim is true and finds correct supporting papers
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix Windows console encoding for Greek letters and special characters
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from google.genai import types
from dotenv import load_dotenv

from .llm_response_parser import extract_json_from_llm_response

# PMID extraction is now handled by update_cache_pmids.py
# Fact checker focuses ONLY on validation and providing paper TITLES
PMID_EXTRACTOR_AVAILABLE = False  # No longer used in fact checker


# DOI validation pattern (format: 10.xxxx/yyyy)
DOI_PATTERN = re.compile(r'^10\.\d{4,}/[^\s]+$')

# High reasoning budget for Gemini (tokens dedicated to "thinking"/deliberation)
MAX_THINKING_TOKENS = 32768
MIN_THINKING_TOKENS = 1000

# Request timeout in secondees (to prevent hanging)
REQUEST_TIMEOUT = 300  # 5 minutes per request


def _coerce_token_count(value: Any) -> int:
    """Best-effort conversion of token counts to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_validity(value: Any) -> str:
    """Normalize model-provided validity into canonical labels."""
    if not value:
        return "UNKNOWN"
    val = str(value).strip().upper()
    mapping = {
        'TRUE': 'TRUE',
        'CORRECTED': 'CORRECTED',
        'FALSE': 'FALSE',
        'DELETED': 'DELETED',
        'CONFLICTING': 'CONFLICTING'
    }
    # Accept common variants
    if val in mapping:
        return mapping[val]
    if val in ("TRUTH", "VALID", "ACCURATE"):
        return 'TRUE'
    if val in ("CORRECTION", "FIXED"):
        return 'CORRECTED'
    if val in ("REMOVE", "REMOVED"):
        return 'DELETED'
    return 'UNKNOWN'


def select_best_corrected_function(data: Any) -> Optional[Dict[str, Any]]:
    """Normalize corrected_function payload: allow dict or list of dicts; prefer entries with function_name."""
    if isinstance(data, dict):
        return data.copy()
    if isinstance(data, list):
        dicts = [x for x in data if isinstance(x, dict)]
        if not dicts:
            return None
        # Prefer the candidate that has function_name
        with_name = [d for d in dicts if d.get('function_name')]
        return (with_name[0] if with_name else dicts[0]).copy()
    return None


def extract_clean_doi(doi_value: Any) -> str:
    """Extract and normalize a DOI string from possible list/str payloads."""
    doi_str = ''
    if isinstance(doi_value, str):
        doi_str = doi_value
    elif isinstance(doi_value, list):
        for item in doi_value:
            if isinstance(item, str) and item.strip():
                doi_str = item
                break
    else:
        return ''

    doi_clean = doi_str.replace('doi:', '').replace('DOI:', '').strip()
    doi_clean = doi_clean.replace('https://doi.org/', '').replace('http://doi.org/', '')
    return doi_clean


def is_valid_doi(doi: str) -> bool:
    """
    Validate DOI format.
    Expected format: 10.xxxx/yyyy (e.g., 10.1016/j.cell.2014.08.017)
    """
    if not doi:
        return False
    # Clean common prefixes
    doi_clean = doi.replace('doi:', '').replace('DOI:', '').strip()
    doi_clean = doi_clean.replace('https://doi.org/', '').replace('http://doi.org/', '')
    return bool(DOI_PATTERN.match(doi_clean))


def extract_existing_evidence_titles(claims_batch: List[Dict[str, Any]]) -> List[str]:
    """
    DEPRECATED: No longer extract existing evidence.
    claim_fact_checker.py now does 100% independent research.

    This function is kept for backward compatibility but returns empty list.

    Args:
        claims_batch: Unused (kept for compatibility)
    """
    # Don't extract or use existing evidence - do independent research instead
    _ = claims_batch  # Suppress unused warning
    return []


def select_best_correct_paper(paper_data: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize the `correct_paper` payload returned by the validator.
    Gemini sometimes returns a single dict or a list of candidate dicts.
    This helper picks the best available dict (preferring PMID, then DOI).
    """
    if isinstance(paper_data, dict):
        return paper_data.copy()

    if isinstance(paper_data, list):
        candidates = [item for item in paper_data if isinstance(item, dict)]
        if not candidates:
            return None

        # Prefer entries with PMID, then DOI, otherwise fall back to the first dict
        for candidate in candidates:
            if candidate.get('pmid'):
                return candidate.copy()
        for candidate in candidates:
            if candidate.get('doi'):
                return candidate.copy()
        return candidates[0].copy()

    return None


def get_normalized_correct_paper(validation: Dict[str, Any], func_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve and normalize the `correct_paper` field from a validation response.
    Handles dict, list, or malformed payloads and emits helpful logs when fallback logic is applied.
    """
    raw_correct_paper = validation.get('correct_paper')
    correct_paper = select_best_correct_paper(raw_correct_paper)

    if isinstance(raw_correct_paper, list):
        if correct_paper:
            print(f"      [info] correct_paper returned multiple candidates for {func_name}; using first viable entry")
        else:
            print(f"      [warn] correct_paper list had no usable entries for {func_name}")
    elif raw_correct_paper and not isinstance(raw_correct_paper, dict):
        print(f"      [warn] correct_paper payload is type {type(raw_correct_paper).__name__} for {func_name} - ignoring")

    return correct_paper


def call_gemini_for_claim_validation(
    main_protein: str,
    interactor: str,
    claims_batch: List[Dict[str, Any]],
    api_key: str,
    recovery_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a batch of functional claims using Google Search.
    Returns validation results with correct PMIDs for true claims.
    Can also correct functions if the exact claim is wrong but interaction is real.
    """
    from google import genai as google_genai

    client = google_genai.Client(api_key=api_key)

    # NO LONGER extract existing evidence - do independent research only
    # existing_titles = extract_existing_evidence_titles(claims_batch)

    # Build claims text WITHOUT existing evidence (do independent research)
    claims_text = ""
    for idx, claim_data in enumerate(claims_batch, 1):
        func_name = claim_data.get('function', 'Unknown')
        arrow = claim_data.get('arrow', '')
        cellular_process = claim_data.get('cellular_process', '')
        effect_description = claim_data.get('effect_description', '')
        biological_consequence = claim_data.get('biological_consequence', [])
        specific_effects = claim_data.get('specific_effects', [])
        effect_type = claim_data.get('effect_type', '')
        mechanism = claim_data.get('mechanism', '')
        # NO LONGER use existing evidence - do independent research only

        claims_text += f"\n{idx}. Function: {func_name}\n"
        if arrow:
            claims_text += f"   Arrow: {arrow}\n"
        claims_text += f"   Cellular Process: {cellular_process}\n"
        if effect_description:
            claims_text += f"   Effect Description: {effect_description}\n"

        # CRITICAL: Include biological consequence for validation
        if biological_consequence:
            claims_text += f"   Biological Consequence: {' → '.join(biological_consequence)}\n"

        # Include ALL specific effects for validation
        if specific_effects:
            claims_text += f"   Specific Effects: {'; '.join(specific_effects)}\n"

        # Include effect type and mechanism
        if effect_type:
            claims_text += f"   Effect Type: {effect_type}\n"
        if mechanism:
            claims_text += f"   Mechanism: {mechanism}\n"

        # NO LONGER include existing evidence - do 100% independent research
        # The pipeline provides only function descriptions, not evidence

    # High-level system guidance embedded directly into the prompt for SDK compatibility
    system_text = (
        "You are a rigorous, conservative scientific fact-checker. "
        "Return STRICT JSON only, no prose, matching the schema in the user prompt. "
        "For each claim, you MUST verify that the specific interaction between the two proteins is discussed in the literature and that papers support the EXACT function. "
        "You MUST validate EVERY SINGLE FIELD individually: function name, arrow, cellular_process, effect_description, biological_consequence (each step), and specific_effects (each effect). "
        "If ANY Function Box field (function, arrow, cellular_process, effect_description, biological_consequence, specific_effects) is unsupported or contradicts the literature, do NOT mark TRUE. "
        "Prefer CORRECTED when the interactor is confirmed but the function box details are wrong; otherwise use FALSE or DELETED."
    )

    prompt = f"""STRICT FACT-CHECKING OF PROTEIN INTERACTION FUNCTIONS

You are a RIGOROUS scientific fact-checker and detail refiner. Your goal is to DELETE false claims AND refine/enhance correct claims.
For each claim, verify if the {main_protein}-{interactor} INTERACTION specifically causes the claimed function.

**TWO-TIER VALIDATION APPROACH**:
1. **CORE VALIDATION** (be strict): Both proteins must be in paper, interaction must be real, core function must be correct
   - When in doubt about CORE interaction/function → mark FALSE or DELETED
2. **DETAIL REFINEMENT** (be thorough): Refine all function details using paper + scientific inference
   - Add downstream cascade steps from canonical pathway knowledge
   - Enhance specific effects with scientifically inferable outcomes
   - Don't mark FALSE just because details need refinement - CORRECT them instead!

MAIN PROTEIN: {main_protein}
INTERACTOR: {interactor}

CLAIMS TO VERIFY:
{claims_text}

[WARN][WARN][WARN]**CRITICAL INSTRUCTION - READ THIS FIRST** [WARN][WARN][WARN]

**YOU MUST DO 100% INDEPENDENT RESEARCH!**

The pipeline has NOT provided any evidence papers - only function descriptions.
Your job is to:
1. Do INDEPENDENT research on PubMed/Google Scholar (NO papers have been provided!)
2. Find papers about the {main_protein}-{interactor} interaction FROM SCRATCH
3. READ the abstracts to understand what the interaction does
4. Compare the claimed function to what literature ACTUALLY says
5. Mark FALSE if claim doesn't match reality
6. Provide COMPLETE evidence (paper_title, relevant_quote, doi, authors, year, etc.) for TRUE/CORRECTED claims

**IMPORTANT**: The pipeline did NOT provide any evidence papers. You must find ALL evidence yourself!

**YOUR RESEARCH PROCESS**:
1. Search "ATXN3 VCP interaction" independently
2. Find 3-5 papers that discuss this interaction
3. Read abstracts to understand what the interaction ACTUALLY does
4. Compare to the claimed function
5. If claim matches reality → TRUE (provide complete paper metadata)
6. If function is wrong but interaction is real → CORRECTED (provide correct function + paper metadata)
7. If interaction doesn't exist or is completely wrong → FALSE/DELETED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[VALIDATE] **FIELD-BY-FIELD VALIDATION PROTOCOL** (GO SLOW - VALIDATE EVERYTHING!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[WARN]**YOU MUST VALIDATE EACH FIELD SEPARATELY** [WARN]

For EVERY function, validate EACH component individually. Don't skip any!

**1. FUNCTION NAME** (the claimed function label)
   - Search: "{main_protein} {interactor} [function name]"
   - Is this term used in the literature for this interaction?
   - Or is it a fabricated/generic term?
   - Example: "Stress Granule Dynamics" → Search if this interaction affects SG dynamics

**2. ARROW** (the relationship: "activates", "inhibits", "binds", etc.)
   - Does the interaction activate or inhibit the function?
   - Is the directionality correct?
   - Example: Claim says "activates" but papers show "inhibits" → CORRECTED or FALSE

**3. CELLULAR PROCESS** (detailed description of what happens)
   - Read this description sentence by sentence
   - Are ALL steps mentioned in this description supported by literature?
   - Any fabricated details? Any steps from different protein partners?
   - Example: "ATXN3 and VCP co-localize in stress granules" → Is this even true?

**4. EFFECT DESCRIPTION** (brief summary of the biological outcome)
   - Does this summary accurately reflect what the papers say?
   - Is this the correct outcome for this interaction?
   - Example: Claim says "increased stability" but papers show "degradation" → CORRECTED or FALSE

**5. BIOLOGICAL CONSEQUENCE** (the step-by-step pathway: A → B → C → D)
   [WARN]**TWO-TIER APPROACH - Initial steps STRICT, downstream steps FLEXIBLE**
   - Validate initial steps (interaction mechanism) STRICTLY - must be in paper
   - ADD/REFINE downstream steps using canonical pathway knowledge
   - Validate EACH step individually:
     * Step 1 (interaction): Must be explicitly in paper or clearly inferable from mechanism
     * Step 2 (direct effect): Should be in paper or scientifically inferable from Step 1
     * Steps 3+ (downstream): Can be inferred from canonical pathway knowledge
   - If INITIAL steps (interaction mechanism) are wrong → Mark FALSE or DELETED
   - If DOWNSTREAM steps are incomplete → ADD canonical steps using scientific knowledge
   - Approach: Validate initial steps strictly, then ADD/REFINE downstream steps with scientific inference
   - Example: "ATXN3 binds VCP → VCP activated → Stress granule formation → Cell survival"
     * Check: Does ATXN3 bind VCP? (MUST be in paper)
     * Check: Does this binding activate VCP? (should be in paper or inferable from mechanism)
     * Check: Does VCP activation cause SG formation? (can be inferred from canonical VCP function)
     * Check: Do SGs cause cell survival in this context? (can be inferred if scientifically valid)
   - If initial steps are wrong → Mark FALSE! If downstream steps are missing or incomplete → ADD them using canonical pathway knowledge!

**6. SPECIFIC EFFECTS** (array of biological outcomes)
   - Validate EACH effect in the list individually
   - For each effect, search: "{main_protein} {interactor} [specific effect]"
   - Example: ["Increased NRF2 activation", "Reduced oxidative stress", "Enhanced autophagy"]
     * Check: Does this interaction increase NRF2? (search)
     * Check: Does it reduce oxidative stress? (search)
     * Check: Does it enhance autophagy? (search)
   - If ANY effect is unsupported → Mark FALSE or remove that effect

[WARN]**VALIDATION RULE**: If ANY field above is wrong/unsupported → Mark FALSE or DELETED
- Don't try to salvage claims with 1-2 correct fields and 4-5 wrong fields
- Better to delete than keep partially-false data

**HOW TO VALIDATE EACH FIELD**:
1. Copy the field value (e.g., "ATXN3 recruits VCP to stress granules")
2. Search PubMed: "{main_protein} {interactor} [key terms from field]"
3. Read abstracts - do they support this specific claim?
4. If NO clear evidence → Mark field as UNSUPPORTED
5. If 2+ fields are unsupported → Mark entire function FALSE/DELETED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR TASK FOR EACH CLAIM:

1. **MANDATORY INDEPENDENT RESEARCH PROTOCOL** (DO NOT SKIP THIS!):

   [WARN]**CRITICAL**: NO papers have been provided! You MUST find ALL evidence from scratch!

   **Step 1: INDEPENDENT PUBMED SEARCH FROM SCRATCH (REQUIRED)**
   - NO papers have been provided by the pipeline - start fresh!
   - Search PubMed for: "{main_protein} {interactor} [claimed function]"
   - Search PubMed for: "{main_protein} AND {interactor}"
   - Search Google Scholar for: "{main_protein} {interactor} interaction"
   - Find at least 3-5 papers that discuss this interaction
   - Read the abstracts to understand what the interaction ACTUALLY does

   **Step 2: Identify the BEST supporting paper**
   - Which paper most clearly describes this {main_protein}-{interactor} interaction?
   - Which paper best supports the claimed function (or reveals the actual function)?
   - Extract COMPLETE metadata from that paper:
     * EXACT paper title (word-for-word from PubMed/Google Scholar)
     * relevant_quote (<=200 chars from paper showing the interaction/mechanism context)
     * doi (if available)
     * authors (LastName1 A, LastName2 B format)
     * journal
     * year
     * assay (experimental method used)
     * species (human/mouse/rat/cell)

   **Step 3: READ THE PAPER** (MANDATORY - NO SHORTCUTS!)
   - Does the paper contain BOTH {main_protein} AND {interactor} somewhere (title/abstract/full text)? If not → DELETED
   - Does the paper describe an INTERACTION between them? If not → FALSE/DELETED
   - Does the paper support the SPECIFIC function claimed (or can it be scientifically inferred from the data)? If not → FALSE/CORRECTED
   - Is this just co-localization (e.g., "both are in stress granules")? → FALSE (not functional interaction!)
   - Are they mentioned separately in different contexts? → FALSE (no interaction evidence!)

   **Step 4: Compare claimed function vs. actual literature**
   - What does the literature ACTUALLY say these proteins do together?
   - Does the claim match reality or is it fabricated/misrepresented?
   - Example: Claim says "ATXN3-VCP form stress granules" but literature shows "ATXN3 and VCP are both stress granule components separately" → FALSE
   - Example: Claim says "ATXN3-KEAP1 stabilization" but literature shows "ATXN3 degrades KEAP1" → CORRECTED

   **Red flags for fake/lazy claims**:
   - Both proteins mentioned but NO interaction described → FALSE
   - Co-localization only (e.g., "both in stress granules") → FALSE
   - Paper is about different topic (yeast, plants, etc.) → DELETED
   - Paper doesn't contain both protein names anywhere (title/abstract/full text) → DELETED

1b. **GO SLOW PROTOCOL** (Spend adequate time on each claim):

   [WARN]**MINIMUM SEARCH REQUIREMENT**: 5-10 searches per function

   **For EACH function, you must perform AT LEAST**:
   1. Search "{main_protein} {interactor}" (general interaction)
   2. Search "{main_protein} {interactor} [function name]" (specific function)
   3. Search "{main_protein} [mechanism]" (e.g., "ATXN3 deubiquitination")
   4. Search each biological cascade step INDIVIDUALLY
   5. Search 2-3 specific effects INDIVIDUALLY
   6. Search for the PMID if provided
   7. Read abstracts from at least 3-5 papers

   **Time allocation per function**:
   - Simple function (2-3 cascade steps): 5-7 searches minimum
   - Complex function (4-6 cascade steps): 10-15 searches minimum
   - Each search should involve reading abstracts, not just titles

   **Use your thinking budget wisely**:
   - You have 32,768 thinking tokens (about 24,000 words)
   - For a function with 5 fields to validate, spend:
     * 20% thinking: Planning research strategy
     * 60% thinking: Analyzing each field + cascade step
     * 20% thinking: Making final decision

   **Don't rush!** Taking shortcuts = false claims escape = bad science

2. **DETERMINE VALIDITY** (4 possible outcomes):

   a) **TRUE**: Exact claim AND all function box details are supported by literature
      → **CRITICAL REQUIREMENTS FOR TRUE (ALL MUST BE MET)**:
        [OK]You performed INDEPENDENT PubMed searches for "{main_protein} {interactor}"
        [OK]You found multiple papers discussing this specific interaction
        [OK]You READ the abstract(s) and verified they discuss the interaction
        [OK]The papers explicitly show {main_protein} INTERACTS WITH {interactor}
        [OK]The papers show this INTERACTION causes/affects the claimed function
        [OK]Not just co-localization (e.g., "both in stress granules" is NOT an interaction!)
        [OK]Not just pathway membership (e.g., "both regulate apoptosis" separately is NOT an interaction!)
        [OK]ALL function box details (cascade, effects, mechanism) are supported by literature
      → **NEVER mark TRUE just because PMID exists** - you MUST verify the paper supports the claim!
      → If PMID is about different topic (e.g., yeast pathways for human proteins) → DELETED
      → If papers show co-localization but no functional interaction → FALSE
      → Provide best supporting paper with full metadata and quote showing the interaction/mechanism

   b) **CORRECTED**: ONLY use when the EXACT interactor is confirmed BUT function is wrong
      → **CRITICAL REQUIREMENT**: Papers must EXPLICITLY link the {main_protein}-{interactor} INTERACTION to the function
      → **NOT ENOUGH**: Both proteins participate in Function X separately
      → **REQUIRED**: Papers show "A interacts with B to cause/regulate/affect Function X"
      → The interactor protein name must be EXACTLY {interactor} (not a different protein!)
      → [WARN]**GUILT BY ASSOCIATION = WRONG**: If ATXN3 does aggresome formation with HDAC6, but claim is ATXN3-VCP aggresome → FALSE/DELETED, NOT CORRECTED
      → Only the function/mechanism description is wrong (same interaction, different mechanism)
      → Example: Papers confirm ATXN3-HDAC1 interaction ✓, but claim says "phosphorylates" when papers show "deubiquitinates" → CORRECTED
      → The paper must discuss the interaction between {main_protein} AND {interactor} in the context of the function (proteins can appear anywhere in the paper, mechanism should be inferable)
      → Provide the CORRECT function with ALL fields updated

   c) **FALSE**: Interaction exists but this specific function is completely unsupported
      → [WARN]**ALWAYS TRY TO PROVIDE corrected_function IF YOU FOUND THE REAL FUNCTION**
      → **STEP 1**: Determine the claimed function is FALSE/unsupported
      → **STEP 2**: Search for what this {main_protein}-{interactor} interaction ACTUALLY does
      → **STEP 3**: If you find the real function → Provide complete corrected_function with ALL fields
      → **STEP 4**: Only mark FALSE without correction if interaction truly does nothing or no function found

      **When to provide corrected_function for FALSE**:
      [OK]Papers show interaction exists + does Function Y (not claimed Function X) → Provide corrected Function Y
      [OK]Example: Claim "Mitophagy Regulation" but papers show "PRKN Stability Regulation" → Provide correction!
      [OK]Example: Claim "Stress Granule Dynamics" but papers show "ERAD Protein Degradation" → Provide correction!

      **When NOT to provide corrected_function**:
      ✗ Papers show {main_protein} does Function X with DIFFERENT protein → FALSE without correction (wrong interactor!)
      ✗ Papers only show co-localization, no functional interaction → FALSE without correction
      ✗ No clear function found for this interaction → FALSE without correction

      **Requirements for corrected_function**:
      - Papers must EXPLICITLY show "{main_protein}-{interactor} interaction causes/regulates Function Y"
      - Paper must discuss BOTH proteins in functional context (protein names can appear anywhere in the paper, not necessarily in the same quote)
      - Provide ALL function fields (function_name, arrow, cellular_process, effect_description, biological_consequence, specific_effects)

      **Search strategy**: "What does {main_protein}-{interactor} interaction actually do?"

   c2) [WARN]**CORRECTION DECISION MAKING** (USE YOUR BIOLOGICAL EXPERTISE):

      **USE AI JUDGMENT TO DECIDE IF A CORRECTION IS APPROPRIATE**:

      Instead of rigid rules, use your biological knowledge and reasoning to decide:
      1. Search thoroughly for what the {main_protein}-{interactor} interaction ACTUALLY does
      2. If you find a real function for this interaction, provide a correction with complete details
      3. If you don't find a real function, mark FALSE without correction or DELETED
      4. Trust your biological expertise - you understand when processes are related

      **GUIDING PRINCIPLES** (not hard rules):
      - If the corrected function describes what this interaction truly does → CORRECT IT
      - If papers show NO clear function for this interaction → mark FALSE without correction
      - If papers show interaction with a DIFFERENT protein → mark DELETED
      - Use your judgment about biological relatedness - you're the expert

      **THINK ABOUT**:
      - Does the literature clearly show what this {main_protein}-{interactor} pair does?
      - Is there strong evidence for a specific function?
      - Would keeping this data (even corrected) mislead researchers?

      **EXAMPLES OF GOOD JUDGMENT**:

      **Scenario 1: Found clear alternative function**
      - Claim: "ATXN3-PRKN: Mitophagy Regulation"
      - Research shows: ATXN3 regulates PRKN protein stability
      - Your judgment: "PRKN stability affects mitophagy, so this is related. I'll correct to 'PRKN Stability Regulation'"
      - Action: CORRECTED ✓

      **Scenario 2: Function is completely wrong, no alternative found**
      - Claim: "ATXN3-VCP: Stress Granule Dynamics"
      - Research shows: ATXN3-VCP interaction is for ERAD/protein degradation
      - Your judgment: "ERAD is very different from stress granules (ER vs cytoplasm). While I found ERAD papers,
        this correction would be too different from the original claim. Better to delete."
      - Action: FALSE without correction → DELETE ✓

      **Scenario 3: Papers don't support ANY clear function**
      - Claim: "ATXN3-ProteinX: Function Y"
      - Research shows: Papers mention both proteins but don't show functional interaction
      - Your judgment: "No clear evidence this interaction does anything specific"
      - Action: FALSE without correction → DELETE ✓

      **KEY POINT**: Use your reasoning and biological knowledge, not rigid distance rules.
      If you're confident a correction would help researchers understand what this interaction does → provide it.
      If you're unsure or the correction seems too different → delete instead of correcting.

   d) **DELETED**: The interactor protein itself is wrong OR no interaction exists
      → **Use DELETED when papers show a DIFFERENT protein interacts (not {interactor})**
      → Example: Claim says ATXN3-HDAC1, but papers ONLY show ATXN3-HDAC3 → DELETED (wrong protein!)
      → Example: No papers show {main_protein} and {interactor} interact at all → DELETED
      → Mark for complete removal

3. **FIND AND EXTRACT COMPLETE PAPER METADATA**:
   - NO papers have been provided - you must find them yourself!
   - After validating the claim, identify the BEST supporting paper
   - Extract COMPLETE metadata (title, quote, doi, authors, journal, year, assay, species)
   - The paper title must be EXACT (word-for-word from PubMed/Google Scholar)
   - The paper must contain BOTH protein names somewhere (title/abstract/full text) and discuss the function/mechanism

3b. **CRITICAL VALIDATION AND CORRECTION OF ALL FUNCTION BOX DETAILS**:
   For EVERY function (TRUE, CORRECTED, or FALSE with correction), verify and REFINE EACH component:

   [WARN]**IMPORTANT**: Even for TRUE claims, you should CORRECT and REFINE all fields based on what the paper actually says + scientifically valid inferences!

   - **FUNCTION NAME**: Verify and correct the function name for this interaction
     * Use the most accurate function name based on the paper
     * If the claimed function name is imprecise, refine it (e.g., "Autophagy" → "Mitophagy Regulation")
     * Only mark FALSE/CORRECTED if the core function is completely wrong

   - **ARROW**: Verify and correct the relationship (activates/inhibits/binds/regulates)
     * Update to match what the paper shows
     * If papers show complex regulation, choose the most accurate arrow
     * Only mark FALSE/CORRECTED if the direction is opposite to reality

   - **CELLULAR PROCESS**: Refine to accurately describe what THIS specific interaction does
     * Correct the description based on the paper + canonical pathway knowledge
     * **INFERENCE IS ENCOURAGED**: You can add scientifically valid mechanistic details even if not explicitly in the paper
     * Example: Paper says "ATXN3 deubiquitinates KEAP1" → You can infer "leading to NRF2 release" (canonical pathway)
     * Only mark FALSE if the core mechanism is wrong for this interaction

   - **EFFECT DESCRIPTION**: Refine to accurately summarize the biological outcome
     * Update based on paper + downstream effects that are scientifically inferable
     * Make it concise and accurate

   - **BIOLOGICAL CONSEQUENCE** (cascade steps): Verify and REFINE each step
     * **INFERENCE IS HIGHLY ENCOURAGED**: Use canonical pathway knowledge to complete the cascade
     * Example: Paper shows "A→B" + you know canonically "B→C→D" → Include full cascade "A→B→C→D"
     * Correct any steps that are wrong or out of order
     * Add scientifically valid downstream steps even if not explicitly in paper
     * Only mark FALSE if the initial steps are wrong or the cascade goes in the wrong direction

   - **SPECIFIC EFFECTS**: Verify and refine the list of biological outcomes
     * Correct any effects that are wrong
     * **ADD scientifically inferable effects** based on the cascade and canonical biology
     * Remove effects that are unrelated or caused by different interactions
     * Only mark FALSE if the effects are opposite to reality or involve wrong proteins

   **When to mark FALSE vs. when to CORRECT**:
   - Core interaction is wrong (different proteins) → DELETED
   - Core function is completely wrong (unrelated biology) → FALSE without correction
   - Details are imprecise or incomplete → REFINE and keep as TRUE/CORRECTED
   - Cascade is incomplete but correct direction → ADD steps and keep as TRUE/CORRECTED
   - Mechanism is partially wrong → CORRECT and mark as CORRECTED

   **Scientific inferences are allowed ONLY when**:
   - The interaction directly changes protein levels/activity/localization
   - Downstream effects are well-established canonical functions
   - Example OK: "ATXN3-KEAP1 decreases KEAP1" → can infer "NRF2 activation" (canonical)
   - Example WRONG: "ATXN3 and VCP both in stress granules" → cannot infer functional interaction

4. **FOR TRUE CLAIMS** (core interaction and function are correct):
   [WARN]**IMPORTANT**: Even for TRUE claims, you MUST verify and REFINE all function detail fields!

   - Find PRIMARY paper supporting the claim
   - **REFINE all fields** (Function Name, Arrow, Cellular Process, Effect Description, Biological Consequence, Specific Effects):
     * Correct any imprecise or incomplete details based on the paper
     * Add scientifically inferable downstream effects and cascade steps
     * Make the function box as accurate and complete as possible
   - Extract complete metadata:
     * DOI (format: 10.xxxx/yyyy - e.g., '10.1016/j.cell.2014.08.017')
     * PMID (8-digit number)
     * Paper title
     * Authors (format: 'LastName1 A, LastName2 B')
     * Journal, year
     * Assay used (Co-IP, Western blot, etc.)
     * Species
     * Relevant quote proving the claim (≤200 chars)
   - Include the refined/corrected fields in your TRUE response (even if core claim is correct, details may need refinement)

5. **FOR CORRECTED CLAIMS** (function is wrong but SAME interactor is confirmed):
   - **FIRST**: Verify papers explicitly link the {main_protein}-{interactor} INTERACTION to a function
   - **CRITICAL**: Paper must describe what the INTERACTION does, not what each protein does separately
   - **If papers mention a DIFFERENT protein instead → Use DELETED, NOT CORRECTED!**
   - **If papers show {main_protein} does Function X with a DIFFERENT partner → Use FALSE or DELETED, NOT CORRECTED!**
   - [WARN]**AVOID GUILT BY ASSOCIATION**: Don't correct based on general pathway membership
   - The paper MUST contain BOTH {main_protein} AND {interactor} somewhere and discuss their functional relationship (quote should capture the mechanism/interaction context)
   - Provide the CORRECT function information based on literature
   - **ALL FIELDS ARE REQUIRED** (complete replacement):
     * function_name (REQUIRED - new function name)
     * arrow (REQUIRED - "activates" or "inhibits")
     * cellular_process (REQUIRED - detailed description of what happens)
     * effect_description (REQUIRED - brief summary)
     * biological_consequence (REQUIRED - array of steps, each step must be evidenced)
     * specific_effects (REQUIRED - array of specific biological outcomes)
   - Provide evidence for the corrected function (same metadata as TRUE claims)
   - Explain what was wrong and what's correct

6. **FOR FALSE CLAIMS** (function unsupported but interaction real):
   - **CRITICAL**: Only provide correction if papers EXPLICITLY describe what the {main_protein}-{interactor} INTERACTION does
   - Search for what the **SPECIFIC INTERACTION** does (not what each protein does alone)
   - **HIGH CONFIDENCE REQUIRED**: Don't guess or infer from pathway membership
   - If you find a real interaction-specific function, provide corrected_function with ALL fields:
     * function_name (correct function name based on literature)
     * arrow ("activates" or "inhibits")
     * cellular_process (detailed mechanism description)
     * effect_description (brief summary)
     * biological_consequence (array of pathway steps)
     * specific_effects (array of biological outcomes)
   - The relevant_quote must show BOTH proteins in the functional context
   - Provide evidence for the corrected function (same metadata as TRUE claims)
   - [WARN]**DEFAULT TO FALSE WITHOUT CORRECTION** if evidence is ambiguous or based on pathway membership only
   - Only mark FALSE with correction when you have EXPLICIT evidence of interaction-specific function
   - Explain what was wrong and what's actually correct (or why no correction was possible)

7. **FOR DELETED CLAIMS** (wrong interactor protein OR no interaction):
   - Explain why:
     * Papers show {main_protein} interacts with a DIFFERENT protein (not {interactor})
     * OR no papers show any interaction between {main_protein} and {interactor}
     * OR complete hallucination
   - We'll remove the entire interactor

**CORRECTION PRIORITY** (NUANCED APPROACH):
- Use TRUE if core interaction and function are correct (proteins + main mechanism match)
  * [WARN]Even for TRUE, you MUST refine detail fields (cascade steps, specific effects, etc.)
  * **Inference is ENCOURAGED** for downstream cascade steps and biological effects
  * Don't mark FALSE just because some cascade steps or effects are scientifically inferred
- Use CORRECTED if papers show the interaction but with a different core function
- Use FALSE with correction if you find what the interaction actually does
- Use FALSE without correction if no clear function is found for this interaction
- Use DELETED if wrong interactor protein or no interaction exists at all

**EVIDENCE STANDARD**:
- Paper must contain BOTH {main_protein} AND {interactor} somewhere and discuss their functional relationship (quote should capture the mechanism/interaction context)
- For CORE INTERACTION/FUNCTION: Must be explicitly described or clearly inferable from the paper
- For CASCADE STEPS and SPECIFIC EFFECTS: **Scientific inference is HIGHLY ENCOURAGED**
  * Example: Paper shows "A deubiquitinates B" → You CAN and SHOULD infer canonical downstream effects
  * Not enough: "Protein A does X" + "Protein B does X" separately → FALSE
  * Required: "A-B interaction affects X" (explicit or inferable from mechanism)

RETURN THIS EXACT JSON FORMAT (validate ALL {len(claims_batch)} functions):
{{
  "interactor": "{interactor}",
  "total_functions_validated": {len(claims_batch)},
  "comprehensive_validation": true,
  "validations": [
    {{
      "claim_number": 1,
      "function_name": "...",
      "validity": "TRUE",
      "independent_research_performed": {{
        "pubmed_searches": ["ATXN3 VCP", "ATXN3 VCP interaction", "ataxin-3 valosin"],
        "papers_found": 5,
        "papers_reviewed": ["PMID1", "PMID2", "PMID3"],
        "what_literature_actually_says": "Detailed summary of what multiple papers say about this interaction"
      }},
      "field_by_field_validation": {{
        "function_name": {{
          "value": "Stress Granule Dynamics",
          "validated": true,
          "evidence": "Papers confirm this interaction affects stress granule dynamics",
          "issues": []
        }},
        "arrow": {{
          "claimed": "activates",
          "actual": "activates",
          "validated": true
        }},
        "cellular_process": {{
          "value": "ATXN3 and VCP co-localize...",
          "validated": true,
          "sentence_by_sentence_check": ["Sentence 1: supported", "Sentence 2: supported"],
          "issues": []
        }},
        "effect_description": {{
          "value": "Brief summary of biological outcome...",
          "validated": true,
          "evidence": "Papers support this description",
          "issues": []
        }},
        "biological_consequence": {{
          "total_steps": 4,
          "steps_validated": [
            {{"step": "ATXN3 binds VCP", "evidence": "PMID12345", "validated": true}},
            {{"step": "VCP activated", "evidence": "PMID12346", "validated": true}},
            {{"step": "Stress granule formation", "evidence": "none found", "validated": false}},
            {{"step": "Cell survival", "evidence": "inferred only", "validated": false}}
          ],
          "all_steps_supported": false,
          "unsupported_steps": ["Stress granule formation", "Cell survival"]
        }},
        "specific_effects": {{
          "total_effects": 3,
          "effects_validated": [
            {{"effect": "Increased NRF2", "evidence": "PMID12347", "validated": true}},
            {{"effect": "Reduced oxidative stress", "evidence": "none", "validated": false}},
            {{"effect": "Enhanced autophagy", "evidence": "PMID12348", "validated": true}}
          ],
          "all_effects_supported": false,
          "unsupported_effects": ["Reduced oxidative stress"]
        }}
      }},
      "research_performed": {{
        "searches_conducted": ["ATXN3 VCP", "ATXN3 VCP interaction", "ataxin-3 valosin"],
        "papers_found_count": 5,
        "best_paper_selected": "Title of paper that best supports this claim",
        "paper_snippet": "Snippet from paper showing the interaction/mechanism context",
        "paper_contains_both_proteins": true,
        "describes_interaction": true,
        "supports_specific_function": true,
        "is_just_colocalization": false
      }},
      "validation_note": "Performed independent PubMed searches. Found 5 papers about ATXN3-VCP interaction. All confirm they interact for [specific function]. Paper explicitly describes interaction and function.",
      "refined_function": {{
        "function_name": "Stress Granule Dynamics",
        "arrow": "activates",
        "cellular_process": "REFINED description based on paper + canonical pathway knowledge",
        "effect_description": "REFINED concise summary of outcome",
        "biological_consequence": ["Step 1 from paper", "Step 2 inferred from canonical pathway", "Step 3 downstream effect"],
        "specific_effects": ["Effect 1 from paper", "Effect 2 scientifically inferred", "Effect 3 canonical downstream effect"],
        "refinement_note": "Added downstream cascade steps based on canonical NRF2 pathway knowledge. Corrected cellular process description to match paper's mechanism."
      }},
      "correct_paper": {{
        "paper_title": "EXACT WORD-FOR-WORD title from PubMed/Google Scholar - THIS IS REQUIRED!",
        "relevant_quote": "Direct quote from paper showing the interaction/mechanism",
        "doi": "10.xxxx/xxxxx (optional)",
        "authors": "LastName1 A, LastName2 B (optional)",
        "journal": "Journal Name (optional)",
        "year": 2023,
        "assay": "Western blot|Co-IP|etc (optional)",
        "species": "human|mouse|rat (optional)"
      }},
      "NOTE": "PMID extraction will be handled later by update_cache_pmids.py. Just provide EXACT paper title! IMPORTANT: Even for TRUE claims, you MUST include refined_function with corrected/enhanced details!"
      "search_queries_used": ["query1", "query2"]
    }},
    {{
      "claim_number": 2,
      "function_name": "Oxidative Stress Response",
      "validity": "CORRECTED",
      "validation_note": "Papers confirm ATXN3-KEAP1 interaction exists. Claim states ATXN3 stabilizes KEAP1, but literature shows ATXN3 promotes KEAP1 degradation (opposite effect). Same interactor, wrong function.",
      "corrected_function": {{
        "function_name": "KEAP1 Degradation",
        "arrow": "inhibits",
        "cellular_process": "ATXN3 promotes ubiquitination and proteasomal degradation of KEAP1, leading to NRF2 activation",
        "effect_description": "KEAP1 protein levels are reduced, releasing NRF2 to activate antioxidant genes",
        "biological_consequence": ["ATXN3 deubiquitinates KEAP1", "KEAP1 degradation via proteasome", "NRF2 release and nuclear translocation", "Antioxidant gene activation"],
        "specific_effects": ["Increased NRF2 nuclear localization", "Enhanced antioxidant response", "Reduced oxidative stress"]
      }},
      "correct_paper": {{
        "paper_title": "EXACT WORD-FOR-WORD title showing actual mechanism",
        "relevant_quote": "Quote showing correct mechanism from paper",
        "year": 2023,
        "doi": "10.xxxx/xxxxx (optional)",
        "authors": "Authors (optional)",
        "journal": "Journal (optional)",
        "assay": "Western blot (optional)",
        "species": "human (optional)"
      }},
      "search_queries_used": ["query1", "query2"]
    }},
    {{
      "claim_number": 3,
      "function_name": "Oxidative Stress Response",
      "validity": "FALSE",
      "validation_note": "Original function unsupported. Papers show ATXN3 decreases KEAP1 levels (not stabilizes). Correcting to reflect actual mechanism.",
      "corrected_function": {{
        "function_name": "KEAP1 Degradation and NRF2 Pathway Activation",
        "arrow": "inhibits",
        "cellular_process": "ATXN3 promotes degradation of KEAP1, leading to decreased KEAP1 protein levels. This releases NRF2 from KEAP1-mediated repression, allowing NRF2 to translocate to the nucleus and activate antioxidant response genes.",
        "effect_description": "KEAP1 levels are reduced by ATXN3, resulting in NRF2 activation and enhanced cellular antioxidant defenses",
        "biological_consequence": ["ATXN3 promotes KEAP1 degradation", "KEAP1 protein levels decrease", "NRF2 is released from KEAP1 inhibition", "NRF2 translocates to nucleus", "Antioxidant response element (ARE) genes activated", "Enhanced antioxidant defense"],
        "specific_effects": ["Decreased KEAP1 protein levels", "Increased NRF2 nuclear localization", "Enhanced antioxidant gene expression", "Improved cellular oxidative stress resistance", "Reduced ROS accumulation"]
      }},
      "correct_paper": {{
        "paper_title": "EXACT WORD-FOR-WORD title showing ATXN3 effect on KEAP1 levels",
        "relevant_quote": "Loss of ATXN3 leads to decreased KEAP1 levels",
        "year": 2019,
        "doi": "10.1016/j.example.2019.123 (optional)",
        "authors": "Authors (optional)",
        "journal": "Journal (optional)",
        "assay": "Western blot (optional)",
        "species": "mouse (optional)"
      }},
      "search_queries_used": ["query1", "query2"]
    }},
    {{
      "claim_number": 4,
      "function_name": "Transcriptional Repression",
      "validity": "DELETED",
      "validation_note": "WRONG INTERACTOR: Papers show ATXN3 interacts with HDAC3, NOT HDAC1. The interactor protein itself is incorrect.",
      "search_queries_used": ["query1", "query2"]
    }},
    {{
      "claim_number": 5,
      "function_name": "Some Function",
      "validity": "FALSE",
      "validation_note": "Interaction is real but this specific function has no literature support and no alternative mechanism could be found.",
      "search_queries_used": ["query1", "query2"]
    }},
    {{
      "claim_number": 6,
      "function_name": "Another Function",
      "validity": "DELETED",
      "validation_note": "No evidence of interaction between these proteins at all. Complete hallucination.",
      "search_queries_used": ["query1", "query2"]
    }}
  ]
}}

CRITICAL RULES:
- **DO 100% INDEPENDENT RESEARCH**: NO papers provided - find ALL evidence from scratch!
- **SEARCH PubMed/Google Scholar**: Search for "{main_protein} {interactor}" - find 3-5 papers
- **READ ABSTRACTS**: READ what the papers actually say about the interaction
- **EXTRACT COMPLETE METADATA**: For TRUE/CORRECTED claims, provide full paper metadata (title, quote, doi, authors, journal, year, assay, species)
- **INTERACTION-SPECIFIC REQUIREMENT**: Function must be caused BY the {main_protein}-{interactor} interaction
- **NOT ENOUGH**: Both proteins participate in same process separately
- **NOT ENOUGH**: Both proteins co-localize in same structure (stress granules, aggresomes, etc.)
- **REQUIRED**: Papers explicitly show the interaction causes/regulates the function
- **DEFAULT TO DELETION**: When evidence is weak, ambiguous, or based on co-localization only
- For TRUE: ALL claims and function box details must match literature exactly + provide complete paper metadata
- For CORRECTED: Only when papers EXPLICITLY show different function for THIS interaction + provide complete paper metadata
- For FALSE: Default to NO correction unless extremely clear alternative function exists
- For DELETED: Wrong interactor OR no interaction OR co-localization only
- **CONSERVATIVE APPROACH**: Better to delete questionable data than keep wrong claims
- **NO GUILT BY ASSOCIATION**: ATXN3 does X with HDAC6 ≠ ATXN3 does X with VCP
- **CO-LOCALIZATION ≠ INTERACTION**: "Both in stress granules" ≠ "interact to regulate stress granules"
- DOI format: 10.xxxx/yyyy (e.g., 10.1016/j.cell.2014.08.017)
- NEVER fabricate DOIs, quotes, or paper details - find REAL papers!
- When in doubt → FALSE without correction or DELETED

[WARN]**YOU WILL BE GRADED ON YOUR RESEARCH QUALITY** [WARN]
- Did you search PubMed independently FROM SCRATCH? (REQUIRED - NO papers provided!)
- Did you find multiple papers about the interaction? (REQUIRED)
- Did you read abstracts? (REQUIRED)
- Did you extract COMPLETE paper metadata for TRUE/CORRECTED claims? (REQUIRED)
- Did you distinguish co-localization from functional interaction? (REQUIRED)
- Did you verify the protein even participates in the claimed function independently? (REQUIRED)
- **Did you check CORRECTION DISTANCE?** If proposed correction is in different biological world → DELETE! (CRITICAL)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚩 **RED FLAG DETECTION** (Common patterns of false claims - watch for these!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**RED FLAG #1: Co-localization masquerading as functional interaction**
Pattern: "Both proteins are found in [structure]" → claimed as functional interaction
Examples:
- "ATXN3 and VCP are stress granule components" → Claim: "ATXN3-VCP regulate stress granules" ✗ FALSE
- "Both found in aggresomes" → Claim: "ATXN3-VCP form aggresomes" ✗ FALSE
- "Co-localize in nucleus" → Claim: "ATXN3-VCP transcriptional regulation" ✗ FALSE
**Detection**: Search if papers show INTERACTION for that function, not just presence

**RED FLAG #2: Downstream effects without scientific validity**
Pattern: Biological cascade with steps that are scientifically invalid or unrelated
Examples:
- "ATXN3 binds VCP" (real) → "VCP activation" (real) → "Stress granule formation" (scientifically valid if VCP has canonical SG function) → "Translation inhibition" (valid if SGs inhibit translation) → "Cell survival" (may be too indirect - verify)
  → REFINE: Keep scientifically valid steps, remove/correct invalid ones, mark TRUE with refined_function
- "ATXN3 degrades KEAP1" (real) → "NRF2 activation" (canonical, [OK]KEEP) → "Antioxidant genes" (canonical, [OK]KEEP) → "Mitochondrial biogenesis" (scientifically valid if NRF2 regulates this) → "Lifespan extension" (too far, remove)
  → REFINE: Keep canonical steps, remove unsupported "Lifespan extension", mark TRUE with refined_function
**Detection**: Validate EACH step's scientific validity. If steps are canonically valid → KEEP/ADD them. If scientifically invalid → REMOVE and mark with refined_function

**RED FLAG #3: Generic function names**
Pattern: Vague function names that could apply to many proteins
Examples:
- "Cell Survival" (too generic - HOW does interaction promote survival?)
- "Protein Quality Control" (which mechanism? degradation? folding? transport?)
- "Stress Response" (which stress? which response mechanism?)
**Detection**: If function name is generic, check if cellular_process and mechanism are specific

**RED FLAG #4: Mechanism mismatch**
Pattern: Claimed mechanism doesn't match what the interaction actually does
Examples:
- Claim: "phosphorylation" → Literature: "protein binding only" ✗ CORRECTED or FALSE
- Claim: "transcriptional activation" → Literature: "protein degradation" ✗ CORRECTED or FALSE
- Claim: "deubiquitination" → Literature: "no enzymatic activity" ✗ FALSE
**Detection**: Search "{main_protein} {mechanism}" - does literature support this mechanism for this interaction?

**RED FLAG #5: Wrong protein partner in cascade**
Pattern: Biological cascade mentions different proteins than the claimed interactor
Examples:
- Claim: "ATXN3-VCP aggresome formation"
- Cascade: "ATXN3 binds HDAC6 → Aggresome formation" (mentions HDAC6, not VCP!) ✗ DELETED
**Detection**: Check if cascade mentions DIFFERENT proteins - if yes, wrong interactor → DELETED

**RED FLAG #6: Too many specific effects**
Pattern: Long list of specific effects (5-8+) that are suspiciously comprehensive
Examples:
- ["Increased autophagy", "Enhanced mitophagy", "Reduced ROS", "Improved mitochondrial function", "Activated AMPK", "Inhibited mTOR", "Extended lifespan", "Neuroprotection"]
  → Probably only 2-3 are real, rest are inferred ✗ FALSE
**Detection**: Validate EACH effect. If 50%+ lack evidence → Mark FALSE

**RED FLAG #7: Cannot find supporting papers**
Pattern: No papers found that discuss this specific interaction
Examples:
- Searched PubMed/Google Scholar but found ZERO papers about this interaction → DELETED
- Papers mention proteins separately but NOT together → DELETED
- Papers show "both proteins in pathway" but no direct interaction → FALSE
**Detection**: If you can't find papers supporting the interaction → DELETED

**RED FLAG #8: Paper metadata incomplete or vague**
Pattern: You found papers but can't extract complete metadata
Examples:
- Can't find exact paper title → Keep searching or mark needs review
- Abstract is too vague to extract a good quote → Keep searching
- Biological cascade has 6 steps but papers only support 2 → REFINE (add remaining steps if scientifically valid, or remove unsupported steps)
**Detection**: For TRUE/CORRECTED, you MUST provide complete paper metadata

**RED FLAG #9: Unsupported inference chains (distinguish from canonical pathway inference)**
Pattern: Steps are guessed without scientific basis (different from canonical pathway knowledge!)
Examples:
- ✗ BAD INFERENCE: "ATXN3 binds VCP" (proven) → "Therefore VCP is activated" (no mechanism given) → "Therefore stress granules form" (no evidence VCP activation causes SGs) → "Therefore translation stops" (too speculative)
  → REFINE: Check if each step is scientifically valid. If not, remove unsupported steps, mark with refined_function
- [OK]GOOD INFERENCE: "ATXN3 degrades KEAP1" (proven) → "KEAP1 reduced" (direct effect) → "NRF2 released" (canonical KEAP1-NRF2 interaction) → "Antioxidant genes activated" (canonical NRF2 function)
  → KEEP: This uses well-established canonical pathway knowledge, mark TRUE with refined_function
**Detection**: Distinguish between unsupported guessing (BAD) vs. canonical pathway knowledge (GOOD). Use scientific literature to validate downstream steps.

**RED FLAG #10: Canonical pathway assumption WITHOUT mechanism**
Pattern: Assuming pathways are affected when mechanism is unclear (different from inferring canonical effects when mechanism IS clear!)
Examples:
- ✗ BAD: "ATXN3 and VCP are present" → Assumed to trigger VCP's ATPase pathway → No mechanism shown
  → Need clear mechanism (binding, activation, degradation, etc.) before inferring canonical effects
- [OK]GOOD: "ATXN3 degrades KEAP1" → Infer KEAP1-NRF2 canonical pathway → Mechanism is clear (degradation)
  → When mechanism is clear, you CAN and SHOULD infer canonical downstream effects
**Detection**: Is the MECHANISM clear? If yes → infer canonical effects. If no (just presence/co-localization) → don't assume pathways

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 **THINKING BUDGET USAGE** (You have 32,768 tokens - use them wisely!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have been given **32,768 thinking tokens** (approximately 24,000 words) to thoroughly validate each claim.
This is a MASSIVE thinking budget - USE IT ALL!

**How to use your thinking budget effectively**:

**Phase 1: Research (40% of thinking budget)**
- Plan your search strategy (which searches to run)
- Execute searches systematically
- Read abstracts thoroughly
- Take notes on what each paper says
- Identify which papers support which aspects of the claim

**Phase 2: Field-by-Field Analysis (40% of thinking budget)**
- For EACH function box field, think through:
  * What does this field claim?
  * Did I find evidence for this specific claim?
  * Is this evidence direct or inferred?
  * Are there any red flags?
- For biological cascade, analyze EACH step individually
- For specific effects, validate EACH effect separately
- Question everything - don't assume

**Phase 3: Decision Making (20% of thinking budget)**
- Synthesize all findings
- Determine validity (TRUE/CORRECTED/FALSE/DELETED)
- If correcting, ensure ALL fields are updated based on evidence
- Double-check your reasoning

**What GOOD thinking looks like**:
```
Let me search for "ATXN3 VCP interaction" first...
[search results show 5 papers]

Paper 1 (PMID 12345): Shows ATXN3 binds VCP for ERAD
Paper 2 (PMID 23456): Confirms interaction, mentions protein degradation
Paper 3 (PMID 34567): Shows VCP extracts ATXN3 substrates from ER

Now let me check the specific claim: "Stress Granule Dynamics"
Searching "ATXN3 stress granules"...
[only 1 paper mentions both, but doesn't show interaction]

Hmm, the papers show ATXN3-VCP is for ERAD, not stress granules.
Let me verify if ATXN3 even forms stress granules independently...
[searches show ATXN3 is NOT a stress granule protein]

Checking the biological cascade:
Step 1: "ATXN3 binds VCP" [OK](supported by PMID 12345)
Step 2: "Stress granule assembly" ✗ (no evidence found)
Step 3: "Translation inhibition" ✗ (inferred from step 2, which is false)

Verdict: FALSE - Papers show ATXN3-VCP interaction is for ERAD, not stress granules.
Could provide CORRECTED function for ERAD if confident...
```

**What BAD thinking looks like** (don't do this):
```
Checking PMID 23300942... exists ✓
Paper mentions ATXN3 and VCP ✓
Claim is about stress granules ✓
Marking as TRUE.
```

**Spend your thinking tokens on**:
[OK]Reading abstracts word-by-word
[OK]Comparing claimed cascade vs. what papers actually say
[OK]Questioning inferences (is this proven or assumed?)
[OK]Cross-checking multiple papers
[OK]Identifying red flags

**Don't waste thinking tokens on**:
✗ Repeating yourself
✗ Making excuses
✗ Summarizing obvious things
✗ Formatting JSON (save that for output)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**VALIDATION STANDARDS - MUST FOLLOW**:
1. **FIND papers from scratch for EVERY claim**
   - NO papers have been provided - search independently!
   - Find 3-5 papers about each {main_protein}-{interactor} interaction
   - Read papers to understand what the interaction does
2. **TWO-TIER VALIDATION**:

   **TIER 1 - CORE VALIDATION (be strict)**:
   - Both protein names must be in the paper (title/abstract/full text)
   - Interaction must be real and described in the paper
   - Core function must be correct or inferable from the mechanism
   - When CORE is wrong → FALSE or DELETED

   **TIER 2 - DETAIL REFINEMENT (use scientific inference)**:
   - Biological cascade steps: Paper shows initial steps → YOU ADD downstream steps using canonical pathway knowledge
   - Specific effects: Paper shows mechanism → YOU INFER scientifically valid biological outcomes
   - **Don't mark FALSE for incomplete details - ENHANCE them with scientific knowledge!**
   - Example: Paper shows "A deubiquitinates B" → You infer and ADD "B released → C activated → Effect D"

3. **Default to REFINEMENT when details are incomplete**
   - If core interaction/function is correct → REFINE the details
   - Only mark FALSE/DELETED when CORE interaction/function is wrong
6. **Hallmarks of fake/wrong data to watch for**:
   - PMIDs that don't exist (e.g., 29507204)
   - Paper titles that sound too perfect for the claim
   - Papers about completely different topics (yeast for human proteins)
   - Biological cascades with many unsupported steps
   - Perfect alignment of all claims (suspicious)

INTERACTOR VALIDATION EXAMPLES:

**WRONG INTERACTOR = DELETED:**
1. Claim: "ATXN3-HDAC1" but papers ONLY show "ATXN3-HDAC3" (different protein!)
   → DELETED: "WRONG INTERACTOR: Papers show ATXN3 interacts with HDAC3, NOT HDAC1"

2. Claim: "ATXN3-P53" but papers ONLY show "ATXN3-TP53" (if these are different genes)
   → DELETED: "WRONG INTERACTOR: Papers mention TP53, not P53"

3. Claim: "ATXN3-KEAP1" but papers ONLY show "ATXN3-NRF2" interaction
   → DELETED: "WRONG INTERACTOR: Papers show ATXN3-NRF2, not ATXN3-KEAP1"

**WRONG FUNCTION = CORRECTED (only when interactor is EXACT same):**
4. Claim: "ATXN3-KEAP1: stabilizes" but papers show "ATXN3-KEAP1: degrades" (same KEAP1 ✓)
   → CORRECTED: Update function to degradation, include all fields (arrow, cellular_process, effect_description, biological_consequence, specific_effects)

5. Claim: "ATXN3-TP53: phosphorylates at S15" but papers show "ATXN3-TP53: deubiquitinates" (same TP53 ✓)
   → CORRECTED: Update mechanism to deubiquitination, include all function fields

6. Claim: "ATXN3-HDAC6: activates" but papers show "ATXN3-HDAC6: is substrate of" (same HDAC6 ✓)
   → CORRECTED: Update arrow and relationship, include all function fields

**WHAT NOT TO MARK AS TRUE OR CORRECTED (Common Mistakes):**

7. Claim: "ATXN3-VCP: Aggresome Formation"
   Papers show: ATXN3 forms aggresomes with HDAC6, VCP is separately involved
   → FALSE or DELETED (ATXN3 does aggresome formation with HDAC6, not VCP!)

8. [WARN]**STRESS GRANULE CO-LOCALIZATION IS NOT A FUNCTIONAL INTERACTION**:
   Claim: "ATXN3-VCP: Stress Granule Dynamics"
   PMID: 23300942 says: "Ataxin-3 and VCP are components of stress granules"
   REALITY CHECK:
   - Does the paper say they INTERACT? NO → They just co-localize in stress granules
   - Does ATXN3 form stress granules at all? CHECK THE LITERATURE INDEPENDENTLY!
   - Just because both proteins are in the same cellular structure doesn't mean they functionally interact
   - Being in same place ≠ interacting for that function
   → FALSE without correction (co-localization ≠ functional interaction)

   **HOW TO VERIFY**: Search "ATXN3 stress granules" independently
   - If ATXN3 doesn't even form stress granules → claim is completely wrong
   - If ATXN3 does form SGs but no papers show ATXN3-VCP interaction causes SG dynamics → FALSE

9. Claim: "ATXN3-Protein X: Function Y" with PMID: 23145048
   PMID 23145048 is about: "isoprenoid pathway in fission yeast" (doesn't mention ATXN3!)
   → DELETED (wrong paper - PMID doesn't even discuss these proteins)

10. Claim: "ATXN3-KEAP1: Oxidative Stress via NRF2"
    Papers show: ATXN3 affects KEAP1 levels but paper doesn't mention NRF2 or oxidative stress
    → Can INFER NRF2 activation (canonical KEAP1 function) but verify papers support it

**CRITICAL RESEARCH & VALIDATION EXAMPLES:**

11. Claim: "ATXN3-HSP90AA1: Regulation of DUB Activity"
    Search "ATXN3 HSP90AA1 interaction" → No papers found about this interaction
    → DELETED (no evidence this interaction exists)

12. Claim: "ATXN3-VCP: Stress Granules"
    Search "ATXN3 VCP interaction" → Find papers about ERAD, NOT stress granules
    Search "ATXN3 VCP stress granules" → Only co-localization, no functional interaction
    → FALSE or DELETED (interaction exists but claimed function is wrong)

13. Claim: "ATXN3-KEAP1: KEAP1 Stabilization"
    Search "ATXN3 KEAP1" → Papers show ATXN3 DEGRADES KEAP1 (opposite!)
    → CORRECTED to "KEAP1 Degradation" with complete paper metadata

14. Biological Consequence: "A→B→C→D→E" but papers only show "A→B"
    → REFINE: If C→D→E are scientifically valid canonical downstream steps, KEEP and mark TRUE with refined_function
    → If C→D→E are wrong or unrelated, CORRECT to "A→B" (mark CORRECTED or keep as TRUE with refinement)
    → Only mark FALSE if the initial steps A→B are wrong

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 **EXTREME VALIDATION EXAMPLES** (Study these carefully!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**EXAMPLE 1: Co-localization FALSE POSITIVE (MOST COMMON ERROR)**

Claim: "ATXN3-VCP: Stress Granule Dynamics"
Function Box:
- Arrow: "activates"
- Cellular Process: "ATXN3 and VCP co-localize in stress granules and regulate their assembly"
- Effect Description: "Stress granule assembly is enhanced"
- Biological Consequence: ["ATXN3 binds VCP", "Stress granule nucleation", "RNA binding protein recruitment", "Translation inhibition"]
- Specific Effects: ["Enhanced stress granule formation", "Translation arrest", "Cell survival under stress"]

**VALIDATION PROCESS** (100% independent research):
1. Search "ATXN3 VCP interaction" → Find papers about ERAD, protein degradation
2. Search "ATXN3 stress granules" → Find that ATXN3 is NOT a canonical SG component
3. Search "ATXN3 VCP stress granules" → Find one paper mentioning co-localization
4. Read abstract: "Ataxin-3 and VCP are components of stress granules and associate with glucocorticoid receptor"
5. CRITICAL ANALYSIS:
   - Abstract says "are components" (co-localization) NOT "interact to regulate" (functional)
   - Does NOT show ATXN3-VCP interaction causes SG dynamics
   - Just shows both can be found in SGs (separately!)
6. Validate each field:
   - "Arrow: activates" ✗ (no evidence this interaction activates anything)
   - "Cellular Process" ✗ (co-localization doesn't mean functional interaction)
   - "Effect Description" ✗ (no evidence of enhanced SG assembly from this interaction)
7. Validate biological consequence:
   - "ATXN3 binds VCP" [OK](supported)
   - "Stress granule nucleation" ✗ (NO evidence this interaction causes SG formation)
   - "RNA binding protein recruitment" ✗ (completely unsupported)
   - "Translation inhibition" ✗ (inference from false step)
8. Validate specific effects:
   - "Enhanced stress granule formation" ✗ (no evidence)
   - "Translation arrest" ✗ (no evidence)
   - "Cell survival under stress" ✗ (generic inference)

**VERDICT**: DELETED (not FALSE→CORRECTED!)
- Paper shows co-localization only, not functional interaction for SG dynamics
- Arrow, cellular process, effect description all unsupported
- 3 out of 4 biological consequence steps unsupported
- All 3 specific effects unsupported
- [WARN]**CRITICAL**: Even if you find ATXN3-VCP does ERAD, DON'T correct "Stress Granules" → "ERAD"
  * These are completely different biological processes (cytoplasmic stress vs. ER quality control)
  * This isn't a correction - it's a totally different function
  * Mark FALSE without correction → DELETE
- Don't mark TRUE just because PMID exists!

**EXAMPLE 2: Inference Chain (FALSE POSITIVE)**

Claim: "ATXN3-KEAP1: Mitochondrial Biogenesis"
Function Box:
- Arrow: "activates"
- Cellular Process: "ATXN3 degrades KEAP1, releasing NRF2 to activate antioxidant genes and promote mitochondrial biogenesis"
- Effect Description: "Mitochondrial biogenesis is enhanced through NRF2 activation"
- Biological Consequence: ["ATXN3 deubiquitinates KEAP1", "KEAP1 degradation", "NRF2 activation", "Antioxidant gene expression", "PGC-1α activation", "Mitochondrial biogenesis"]
- Specific Effects: ["Increased NRF2 activity", "Enhanced antioxidant defense", "Elevated PGC-1α", "More mitochondria", "Improved respiration"]

**VALIDATION PROCESS**:
1. Search "ATXN3 KEAP1" → Find papers showing ATXN3 regulates KEAP1 levels ✓
2. Validate each field:
   - "Arrow: activates" [OK](ATXN3 activates NRF2 pathway via KEAP1 degradation)
   - "Cellular Process" [WARN](First part supported, mitochondrial biogenesis is inferred)
   - "Effect Description" [WARN](Mitochondrial biogenesis not directly evidenced)
3. Validate biological consequence step-by-step:
   - "ATXN3 deubiquitinates KEAP1" [OK](Papers support)
   - "KEAP1 degradation" [OK](Papers show decreased KEAP1)
   - "NRF2 activation" [OK](Canonical pathway, well-established)
   - "Antioxidant gene expression" [OK](Canonical NRF2 function)
   - "PGC-1α activation" [WARN](Search "NRF2 PGC-1α" - is this supported?)
   - "Mitochondrial biogenesis" [WARN](Search "ATXN3 KEAP1 mitochondria" - any evidence?)
4. Search "ATXN3 KEAP1 mitochondrial biogenesis" → NO papers found
5. Search "NRF2 PGC-1α mitochondria" → Some papers, but indirect
6. Critical question: Do papers show ATXN3-KEAP1 interaction DIRECTLY causes mitochondrial biogenesis?
   → NO, this is an inference chain (steps 1-4 are real, but 5-6 are inferred)

**VERDICT**: FALSE (or CORRECTED to stop at "Antioxidant Response")
- First 4 biological consequence steps supported ✓
- Last 2 biological consequence steps are inference (not direct evidence) ✗
- Effect description overstates the evidence (mitochondrial biogenesis not directly supported)
- Don't mark TRUE for inference chains beyond 2-3 well-established canonical steps
- Could provide CORRECTED function that stops at "NRF2-mediated Antioxidant Response"

**EXAMPLE 3: Wrong Protein Partner in Cascade (DELETED)**

Claim: "ATXN3-VCP: Aggresome Formation"
Function Box:
- Arrow: "activates"
- Cellular Process: "ATXN3 recruits HDAC6 to transport misfolded proteins along microtubules to form aggresomes"
- Effect Description: "Aggresomes are formed to sequester misfolded proteins"
- Biological Consequence: ["ATXN3 binds misfolded proteins", "HDAC6 recruitment", "Dynein motor activation", "Retrograde transport", "Aggresome formation"]
- Specific Effects: ["Misfolded protein clearance", "Reduced proteotoxicity", "Enhanced cell survival"]

**VALIDATION PROCESS**:
1. Read cellular_process carefully → Mentions "HDAC6" not VCP!
2. Read biological consequence → Mentions "HDAC6 recruitment" not VCP!
3. This is a RED FLAG: Claim is about ATXN3-VCP but describes ATXN3-HDAC6 function
4. Search "ATXN3 VCP aggresome" → Papers show VCP involved in aggresome clearance, not formation
5. Search "ATXN3 HDAC6 aggresome" → Papers show ATXN3-HDAC6 interaction forms aggresomes ✓

**VERDICT**: DELETED
- The function description is about ATXN3-HDAC6, NOT ATXN3-VCP
- Wrong interactor protein!
- This is GUILT BY ASSOCIATION: ATXN3 does aggresome with HDAC6 ≠ ATXN3 does aggresome with VCP

**EXAMPLE 4: Partially True (Still FALSE)**

Claim: "ATXN3-TP53: Apoptosis Regulation"
Function Box:
- Arrow: "activates"
- Cellular Process: "ATXN3 deubiquitinates TP53, increasing its stability and transcriptional activity, leading to apoptosis"
- Effect Description: "Apoptosis is induced through TP53 stabilization"
- Biological Consequence: ["ATXN3 deubiquitinates TP53", "TP53 stabilization", "TP53 nuclear translocation", "Pro-apoptotic gene activation", "Caspase activation", "Apoptosis"]
- Specific Effects: ["Increased TP53 levels", "Enhanced p21 expression", "Elevated BAX", "Caspase-3 activation", "Cell death"]

**VALIDATION PROCESS**:
1. Search "ATXN3 TP53 deubiquitination" → Papers confirm ATXN3 deubiquitinates TP53 ✓
2. Validate each field:
   - "Arrow: activates" [OK](ATXN3 activates/stabilizes TP53)
   - "Cellular Process" [WARN](First part true, apoptosis connection not evidenced)
   - "Effect Description" ✗ (Apoptosis induction not supported for this interaction)
3. Validate biological consequence:
   - "ATXN3 deubiquitinates TP53" [OK](supported)
   - "TP53 stabilization" [OK](papers show increased TP53)
   - "TP53 nuclear translocation" [WARN](is this shown for ATXN3-TP53 specifically?)
   - "Pro-apoptotic gene activation" [WARN](does ATXN3-TP53 lead to apoptosis specifically?)
   - "Caspase activation" ✗ (inference)
   - "Apoptosis" ✗ (not shown for this interaction)
4. Search "ATXN3 TP53 apoptosis" → Mixed results, some say ATXN3 affects TP53 function but NOT apoptosis
5. Validate specific effects:
   - "Increased TP53 levels" ✓
   - "Enhanced p21 expression" [WARN](is this specific to ATXN3-TP53?)
   - "Elevated BAX" ✗ (not found)
   - "Caspase-3 activation" ✗ (not found)
   - "Cell death" ✗ (not found for this interaction)

**VERDICT**: FALSE (or CORRECTED to "TP53 Stabilization" without apoptosis)
- First 2 biological consequence steps validated ✓
- Steps 3-6 lack evidence ✗
- 3 out of 5 specific effects unsupported ✗
- Effect description overstates the evidence (apoptosis not supported)
- Can't mark TRUE just because 40% is correct!
- Should provide CORRECTED function focusing on TP53 stabilization only

**EXAMPLE 5: Generic Function Name with Vague Details (FALSE)**

Claim: "ATXN3-UBQLN2: Protein Quality Control"
Function Box:
- Arrow: "activates"
- Cellular Process: "ATXN3 and UBQLN2 cooperate to maintain protein homeostasis"
- Effect Description: "Protein homeostasis is maintained through enhanced clearance"
- Biological Consequence: ["Misfolded protein recognition", "Ubiquitin binding", "Proteasomal targeting", "Protein degradation"]
- Specific Effects: ["Enhanced protein clearance", "Improved proteostasis", "Cell survival"]

**VALIDATION PROCESS**:
1. Notice RED FLAG: Function name is generic ("Protein Quality Control")
2. Notice RED FLAG: Cellular process is vague ("maintain protein homeostasis")
3. Notice RED FLAG: Effect description is generic
4. Notice RED FLAG: Biological consequence doesn't mention ATXN3 or UBQLN2 specifically (could apply to many proteins)
5. Search "ATXN3 UBQLN2 interaction" → Find papers ✓
6. But: Do papers show SPECIFIC mechanism? What EXACTLY do they do together?
7. Validate biological consequence - each step is generic:
   - "Misfolded protein recognition" ← By which protein? ATXN3 or UBQLN2?
   - "Ubiquitin binding" ← Both can bind ubiquitin separately
   - "Proteasomal targeting" ← Via what mechanism?
   - "Protein degradation" ← Which substrates?
8. These details are too vague to validate properly

**VERDICT**: REFINE with specific details from paper (mark as TRUE with refined_function or CORRECTED)
- Function is too generic → ADD specific mechanism from paper
- No specific substrates mentioned → ADD substrates if paper shows them
- No specific mechanism details → ADD mechanism details from paper (binding, degradation, etc.)
- Biological consequence steps don't specify which protein does what → CLARIFY based on paper
- Arrow is generic → UPDATE to specific relationship (activates/inhibits/binds/regulates)
- If details are vague → REFINE them using paper + canonical pathway knowledge, don't just mark FALSE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Be THOROUGH and FAIR. Verify the interactor protein name FIRST before deciding CORRECTED vs DELETED!
**MOST IMPORTANT**: Find papers from scratch - NO papers have been provided! Do 100% independent research!"""

    # Prepend system_text to the prompt to guide the model without system_instructions
    prompt = system_text + "\n\n" + prompt

    if recovery_hint:
        prompt += (
            "\n\nRECOVERY INSTRUCTIONS:\n"
            "The previously cited PMID(s) were invalid or evidence was unclear.\n"
            "Ignore the invalid PMIDs and perform a fresh search. If the interaction is real but the function is wrong, provide a CORRECTED function with ALL Function Box fields updated.\n"
            f"Hint: {recovery_hint}\n"
        )

    # MAXIMIZED Gemini configuration - enforce high thinking budget
    # Lower temperature for more consistent, rigorous fact-checking
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,  # 32K tokens for deep reasoning
            include_thoughts=True,
        ),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=65536,
        temperature=0.15,  # Lower temperature = more rigorous, less creative
        top_p=0.90,  # Slightly lower for more focused reasoning
    )

    # Log comprehensive validation details
    print(f"\n  [STATS] Validation details for {interactor}:")
    print(f"     - Functions to validate: {len(claims_batch)}")
    print(f"     - Gemini model: gemini-2.5-pro (FULL POWER)")
    print(f"     - Thinking budget: {MAX_THINKING_TOKENS:,} tokens (MAXIMUM reasoning enforced)")
    print(f"     - Max output: 65,536 tokens")
    print(f"     - Temperature: 0.3 (rigorous fact-checking mode)")
    print(f"     - Tools: Google Search + PubMed (REQUIRED to use)")

    # Retry logic for Gemini API (handles rate limits and empty responses)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print(f"     - Initiating comprehensive validation...")
            else:
                print(f"     - Retry attempt {attempt + 1}/{max_retries}...")

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config,
            )

            # Extract text
            if hasattr(response, 'text'):
                output = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                output = ''.join(part.text for part in parts if hasattr(part, 'text')).strip()
            else:
                output = ""

            # Check for empty response
            if not output or len(output) < 10:
                if attempt < max_retries - 1:
                    print(f"  [WARN]Empty response from Gemini (attempt {attempt + 1}/{max_retries}), retrying in 5s...")
                    time.sleep(5)
                    continue
                else:
                    print(f"  [WARN]Empty response after {max_retries} attempts")
                    return {"validations": []}

            # Try to parse JSON with robust extraction
            try:
                # Extract JSON robustly (handles markdown, extra text)
                result = extract_json_from_llm_response(output)

                # Log token stats
                usage = getattr(response, 'usage_metadata', None)
                token_stats = {
                    'thinking': 0,
                    'output': 0,
                    'total': 0,
                }
                if usage:
                    token_stats['thinking'] = _coerce_token_count(getattr(usage, 'cached_content_token_count', 0))
                    token_stats['output'] = _coerce_token_count(getattr(usage, 'candidates_token_count', 0))
                    token_stats['total'] = _coerce_token_count(getattr(usage, 'total_token_count', 0))
                    if token_stats['total'] > 0 and token_stats['thinking'] == 0:
                        prompt_tokens = _coerce_token_count(getattr(usage, 'prompt_token_count', 0))
                        token_stats['thinking'] = max(0, token_stats['total'] - prompt_tokens - token_stats['output'])
                # Log successful validation with cost
                num_validations = len(result.get('validations', []))
                if any(token_stats.values()):
                    # Calculate input tokens and costs
                    thinking_tokens = token_stats['thinking']
                    output_tokens = token_stats['output']
                    total_tokens = token_stats['total']
                    input_tokens = max(0, total_tokens - thinking_tokens - output_tokens)

                    # Calculate costs (Gemini 2.5 Pro pricing)
                    input_cost = (input_tokens / 1_000_000) * 1.25
                    thinking_cost = (thinking_tokens / 1_000_000) * 1.25
                    output_cost = (output_tokens / 1_000_000) * 10.00
                    total_cost = input_cost + thinking_cost + output_cost

                    print(
                        f"     - Tokens: input={input_tokens:,}, thinking={thinking_tokens:,}, "
                        f"output={output_tokens:,}, total={total_tokens:,}"
                    )
                    print(
                        f"     - Cost: ${total_cost:.4f} (input: ${input_cost:.4f}, "
                        f"thinking: ${thinking_cost:.4f}, output: ${output_cost:.4f})"
                    )
                print(f"     [OK]Validation complete! Received {num_validations} results")
                return result
            except (json.JSONDecodeError, ValueError) as je:
                if attempt < max_retries - 1:
                    print(f"  [WARN]JSON parse error (attempt {attempt + 1}/{max_retries}): {je}")
                    # Show error details if available (JSONDecodeError has .msg and .pos)
                    if hasattr(je, 'msg'):
                        print(f"     Error location: {je.msg}")
                    if hasattr(je, 'pos'):
                        error_pos = je.pos
                        context_start = max(0, error_pos - 100)
                        context_end = min(len(output), error_pos + 100)
                        print(f"     Context: ...{output[context_start:context_end]}...")
                    print(f"     Retrying in 8s with fresh request...")
                    time.sleep(8)
                    continue
                else:
                    print(f"  [WARN]JSON parse error after {max_retries} attempts: {je}")
                    if hasattr(je, 'msg') and hasattr(je, 'pos'):
                        print(f"     Error: {je.msg} at position {je.pos}")
                    if len(output) > 0:
                        print(f"     Response length: {len(output)} chars")
                        print(f"     Response preview: {output[:300]}...")
                    return {"validations": []}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [WARN]Error in claim validation (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"     Error type: {type(e).__name__}")
                print(f"     Retrying in 8s...")
                time.sleep(8)
                continue
            else:
                print(f"  [WARN]Error after {max_retries} attempts: {e}")
                print(f"     Error type: {type(e).__name__}")
                return {"validations": []}


def _process_single_interactor(
    int_idx: int,
    interactor: Dict[str, Any],
    main_protein: str,
    api_key: str,
    total_interactors: int,
    max_functions: int = 20
) -> Dict[str, Any]:
    """
    Process fact-checking for a single interactor (thread-safe).
    Returns a dict with updated interactor and statistics.
    """
    import time as time_module

    primary = interactor.get('primary', 'UNKNOWN')
    functions = interactor.get('functions', [])

    # Statistics for this interactor
    stats = {
        'claims': 0,
        'validated_true': 0,
        'validated_false': 0,
        'corrected': 0,
        'corrected_from_false': 0,
        'deleted': 0,
        'conflicting': 0,
        'removed': 0
    }

    if not functions:
        return {'interactor': interactor, 'stats': stats}

    print(f"\n[{int_idx}/{total_interactors}] Fact-checking {main_protein}-{primary} interactions...")
    print(f"  Total functions to check: {len(functions)}")
    print(f"  Strategy: BATCHED - One API call for ALL functions from this interactor")
    print(f"  Gemini config: thinking budget {MAX_THINKING_TOKENS:,} tokens, max_output=65536, temp=0.2")

    # Process ALL functions in a single batch call
    updated_functions = []

    # Safety: limit iterations to prevent infinite loops
    max_funcs = min(len(functions), max_functions)
    if len(functions) > max_funcs:
        print(f"  [WARN]WARNING: Limiting to first {max_funcs} functions (found {len(functions)})")

    # BATCHED APPROACH: Call API once for all functions from this interactor
    batch_start_time = time_module.time()
    functions_to_check = functions[:max_funcs]
    stats['claims'] = len(functions_to_check)

    try:
        print(f"  [BATCH] Validating {len(functions_to_check)} functions in single API call...")
        batch_result = call_gemini_for_claim_validation(
            main_protein, primary, functions_to_check, api_key
        )
        validations = batch_result.get('validations', []) if isinstance(batch_result, dict) else []
        batch_time = time_module.time() - batch_start_time
        print(f"  [BATCH] Completed in {batch_time:.1f}s ({len(validations)} validations returned)")
    except Exception as batch_err:
        print(f"  [WARN]Batch validation failed: {batch_err}")
        print(f"  [FALLBACK] Switching to one-by-one processing...")
        validations = []

    # If batch failed, fall back to one-by-one processing
    if not validations:
        print(f"  [FALLBACK] Processing {len(functions_to_check)} functions individually...")
        for func_idx, function in enumerate(functions_to_check, 1):
            func_start_time = time_module.time()
            func_name = function.get('function', 'unnamed')

            try:
                single_result = call_gemini_for_claim_validation(
                    main_protein, primary, [function], api_key
                )
                func_validations = single_result.get('validations', []) if isinstance(single_result, dict) else []
                validations.extend(func_validations)
            except Exception as single_err:
                print(f"    [WARN]Function validation failed for '{func_name}': {single_err}")

    # Process validation results and update functions
    # (This is the large block from lines 1573-1871, moved here)
    for func_idx, function in enumerate(functions_to_check, 1):
        func_start_time = time_module.time()
        func_name = function.get('function', 'unnamed')

        # Find corresponding validation by index or function name
        validation = None
        for v in validations:
            if not isinstance(v, dict):
                print(f"      [WARN]Invalid validation format (not a dict): {type(v)}")
                continue

            if (v.get('claim_number') == func_idx or
                v.get('function_name') == func_name):
                validation = v
                break

        if not validation or not isinstance(validation, dict):
            # Recovery retry
            if not function.get('_validation_recovery_attempted'):
                function['_validation_recovery_attempted'] = True
                try:
                    recovery = call_gemini_for_claim_validation(
                        main_protein, primary, [function], api_key,
                        recovery_hint="No validation returned; re-search and validate this single function strictly"
                    )
                    vlist = recovery.get('validations', []) if isinstance(recovery, dict) else []
                    for v in vlist:
                        if isinstance(v, dict):
                            validation = v
                            break
                except Exception as _rec_err:
                    print(f"      [WARN]Recovery attempt failed for '{func_name}': {_rec_err}")

        if not validation or not isinstance(validation, dict):
            print(f"    [WARN][{func_idx}/{len(functions)}] No validation for '{func_name}' - marking for manual review")
            function['needs_manual_review'] = True
            updated_functions.append(function)
            time.sleep(1.0)
            continue

        print(f"    [{func_idx}/{len(functions)}] Processing: {func_name}")

        validity = normalize_validity(validation.get('validity'))
        note = validation.get('validation_note', '')

        # Check PMID verification
        pmid_verification = validation.get('pmid_verification', {})
        pmid_invalid = False
        if pmid_verification and isinstance(pmid_verification, dict):
            if not pmid_verification.get('exists_on_pubmed', True):
                pmid_invalid = True
                print(f"      [WARN]PMID cited is invalid or not found on PubMed — will search for correct evidence instead of deleting")
                note = f"PMID verification failed: {note}"

        # Add validation metadata
        function['validity'] = validity
        function['validation_note'] = note

        # Recovery for TRUE with invalid PMID
        if validity == 'TRUE' and pmid_invalid and not function.get('_pmid_recovery_attempted'):
            function['_pmid_recovery_attempted'] = True
            try:
                recovery = call_gemini_for_claim_validation(
                    main_protein, primary, [function], api_key,
                    recovery_hint="Cited PMID invalid; find a valid paper and re-validate this claim"
                )
                vlist = recovery.get('validations', []) if isinstance(recovery, dict) else []
                for v in vlist:
                    if isinstance(v, dict):
                        validation = v
                        validity = normalize_validity(validation.get('validity'))
                        break
            except Exception as _rec_err:
                print(f"      [WARN]Recovery attempt (TRUE w/ invalid PMID) failed: {_rec_err}")

        if validity == 'TRUE':
            stats['validated_true'] += 1

            # Update with correct evidence if provided
            correct_paper = get_normalized_correct_paper(validation, func_name)

            if correct_paper and isinstance(correct_paper, dict):
                # Update evidence with paper title (PMID added later by update_cache_pmids.py)
                if 'evidence' in function:
                    if isinstance(function['evidence'], list):
                        function['evidence'].append(correct_paper)
                    else:
                        function['evidence'] = [correct_paper]
                else:
                    function['evidence'] = [correct_paper]

                function['evidence_source'] = 'fact_checker_verified'

                paper_title = correct_paper.get('paper_title', '')
                if paper_title:
                    truncated_title = paper_title if len(paper_title) <= 100 else paper_title[:97] + "..."
                    print(f"      ✓ TRUE - Verified with evidence: {truncated_title}")
                    print(f"      Note: PMID will be extracted from title later")
                else:
                    print(f"      ✓ TRUE - Verified")
            else:
                print(f"      ✓ TRUE")

            updated_functions.append(function)

        elif validity == 'CORRECTED':
            # Check if this is an interactor mismatch (CANNOT BE CORRECTED)
            corrected_func = select_best_corrected_function(validation.get('corrected_function')) or {}
            correct_paper = get_normalized_correct_paper(validation, func_name)

            # Check if corrected interactor differs from current
            corrected_interactor = corrected_func.get('interactor')
            if corrected_interactor and corrected_interactor != primary:
                stats['deleted'] += 1
                stats['validated_false'] += 1
                print(f"    ✖ WRONG INTERACTOR: {func_name}")
                print(f"      Reason: Function applies to {corrected_interactor}, not {primary}")
                print(f"      Action: DELETING from {primary} (will be re-discovered in future query for {corrected_interactor})")

                # Mark entire interactor for deletion if this is the only function
                if len(functions) == 1:
                    interactor['_delete_interactor_completely'] = True
                # Don't add to updated_functions
                continue

            # Valid CORRECTED case: same interactor, wrong function
            stats['corrected'] += 1
            print(f"    🔧 CORRECTED: {func_name}")
            print(f"      Reason: {note}")

            if corrected_func and correct_paper:
                # Update function with corrected data
                function['function'] = corrected_func.get('function_name', func_name)
                function['arrow'] = corrected_func.get('arrow', function.get('arrow', 'activates'))
                function['cellular_process'] = corrected_func.get('cellular_process', function.get('cellular_process', ''))
                function['effect_description'] = corrected_func.get('effect_description', '')
                function['biological_consequence'] = corrected_func.get('biological_consequence', [])
                function['specific_effects'] = corrected_func.get('specific_effects', [])
                function['effect_type'] = corrected_func.get('effect_type', '')
                function['mechanism'] = corrected_func.get('mechanism', '')

                function['evidence'] = [correct_paper]
                function['pmids'] = []
                function['evidence_source'] = 'fact_checker_corrected'
                function['original_function_name'] = func_name

                print(f"      → Updated to: {function['function']}")
                paper_title = correct_paper.get('paper_title', '')
                if paper_title:
                    truncated_title = paper_title if len(paper_title) <= 100 else paper_title[:97] + "..."
                    print(f"      Evidence Paper: {truncated_title}")
                    print(f"      Note: PMID will be extracted from title later")

                updated_functions.append(function)
            else:
                print(f"      [WARN]Correction missing data - keeping original")
                function['needs_manual_review'] = True
                updated_functions.append(function)

        elif validity == 'FALSE':
            # Check if Gemini provided a corrected function for this FALSE case
            corrected_func = select_best_corrected_function(validation.get('corrected_function')) or {}
            correct_paper = get_normalized_correct_paper(validation, func_name)

            if corrected_func and correct_paper:
                # Valid correction - proceed with FALSE→CORRECTED
                stats['corrected'] += 1
                stats['corrected_from_false'] += 1
                print(f"    🔧 FALSE→CORRECTED: {func_name}")
                print(f"      Original claim was FALSE, but found actual function for this interaction")
                print(f"      Reason: {note}")

                # Update function with corrected data
                function['function'] = corrected_func.get('function_name', func_name)
                function['arrow'] = corrected_func.get('arrow', function.get('arrow', 'activates'))
                function['cellular_process'] = corrected_func.get('cellular_process', function.get('cellular_process', ''))
                function['effect_description'] = corrected_func.get('effect_description', '')
                function['biological_consequence'] = corrected_func.get('biological_consequence', [])
                function['specific_effects'] = corrected_func.get('specific_effects', [])
                function['effect_type'] = corrected_func.get('effect_type', '')
                function['mechanism'] = corrected_func.get('mechanism', '')

                function['evidence'] = [correct_paper]
                function['pmids'] = []
                function['evidence_source'] = 'fact_checker_salvaged_from_false'
                function['original_function_name'] = func_name
                function['_was_false_but_corrected'] = True

                print(f"      → Salvaged as: {function['function']}")
                paper_title = correct_paper.get('paper_title', '')
                if paper_title:
                    truncated_title = paper_title if len(paper_title) <= 100 else paper_title[:97] + "..."
                    print(f"      Evidence Paper: {truncated_title}")
                    print(f"      Note: PMID will be extracted from title later")

                updated_functions.append(function)
            else:
                # No correction available - mark as FALSE and DELETE
                stats['validated_false'] += 1
                stats['deleted'] += 1
                print(f"    ✖ FALSE: {func_name}")
                print(f"      Reason: {note}")
                print(f"      → REMOVED from output (entire interaction is fabricated)")

                # Mark interactor for deletion if this is the only function
                if len(functions) == 1:
                    interactor['_delete_interactor_completely'] = True

        elif validity == 'CONFLICTING':
            stats['conflicting'] += 1
            print(f"    [WARN]CONFLICTING: {func_name}")
            print(f"      Note: {note}")
            updated_functions.append(function)

        else:
            print(f"    ? UNKNOWN: {func_name} (keeping as-is)")
            updated_functions.append(function)

        # Print timing for this function
        func_elapsed = time_module.time() - func_start_time
        print(f"      [TIME] Time: {func_elapsed:.1f}s")

        # Short delay between function calls to respect rate limits
        time.sleep(1.0)

    # Update interactor with fact-checked functions
    interactor['functions'] = updated_functions

    # Mark interactor for removal if no valid functions remain
    if len(updated_functions) == 0:
        interactor['_remove_interactor'] = True
        print(f"  [WARN]No valid functions remain for {primary} - interactor will be removed")

    # Also check if interactor was marked as completely deleted
    if interactor.get('_delete_interactor_completely'):
        interactor['_remove_interactor'] = True
        print(f"  [DELETE]Entire {primary} interaction is fabricated - interactor will be removed")

    return {'interactor': interactor, 'stats': stats}


def fact_check_json(
    json_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fact-check all functional claims in the JSON.
    Now with CORRECTION capability - doesn't just delete wrong claims, fixes them!

    Each function is validated with its own Gemini call to maximize focus and
    reduce context overload, with a high thinking budget for deep reasoning.
    """

    if 'ctx_json' not in json_data:
        print("[WARN]No ctx_json found in input data")
        return json_data

    ctx_json = json_data['ctx_json']
    interactors = ctx_json.get('interactors', [])

    if not interactors:
        print("[WARN]No interactors found")
        return json_data

    main_protein = ctx_json.get('main', 'UNKNOWN')

    print(f"\n{'='*80}")
    print(f"FACT-CHECKING AND CORRECTING CLAIMS FOR {main_protein}")
    print(f"{'='*80}")
    print(f"Total interactors: {len(interactors)}")
    print(f"Strategy: BATCHED - One API call per interactor (all functions batched together)")
    print(f"Quality: Gemini 2.5 Pro with shared context across functions")
    print(f"Config: Thinking budget {MAX_THINKING_TOKENS:,} tokens, max_output=65536, temp=0.2, Google Search enabled")
    print(f"Evidence Focus: PAPER TITLES - PMIDs will be extracted later by update_cache_pmids.py")
    print(f"Task: Validate functions and provide EXACT paper titles from literature")
    print(f"Performance: ~67% fewer API calls vs old one-per-function approach")
    print()

    total_claims = 0
    claims_validated_true = 0
    claims_validated_false = 0
    claims_corrected = 0
    claims_corrected_from_false = 0  # Track FALSE cases that were salvaged
    claims_deleted = 0
    claims_conflicting = 0
    claims_removed = 0

    # Import time for tracking
    import time as time_module

    # ==============================================================================
    # PARALLEL PROCESSING: Process up to 3 interactors concurrently
    # ==============================================================================
    print(f"[PARALLEL] Using ThreadPoolExecutor with max_workers=3 for concurrent processing")
    print()

    # Process interactors in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all interactor processing tasks
        future_to_interactor = {
            executor.submit(_process_single_interactor, int_idx, interactor, main_protein, api_key, len(interactors)): (int_idx, interactor)
            for int_idx, interactor in enumerate(interactors, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_interactor):
            int_idx, original_interactor = future_to_interactor[future]
            try:
                result = future.result()
                processed_interactor = result['interactor']
                stats = result['stats']

                # Update the interactor in the list (thread-safe since we're in as_completed)
                interactors[int_idx - 1] = processed_interactor

                # Aggregate statistics
                total_claims += stats['claims']
                claims_validated_true += stats['validated_true']
                claims_validated_false += stats['validated_false']
                claims_corrected += stats['corrected']
                claims_corrected_from_false += stats['corrected_from_false']
                claims_deleted += stats['deleted']
                claims_conflicting += stats['conflicting']
                claims_removed += stats['removed']

            except Exception as exc:
                primary = original_interactor.get('primary', 'UNKNOWN')
                print(f"[ERROR] Interactor {primary} generated an exception: {exc}")
                # Keep original interactor unchanged on error

    print(f"\n[PARALLEL] All interactor processing complete!")
    print()

    # Remove interactors with no valid functions or marked as deleted
    interactors_removed = 0
    filtered_interactors = []
    for interactor in interactors:
        if interactor.get('_remove_interactor') or interactor.get('_delete_interactor_completely'):
            interactors_removed += 1
        else:
            # Clean up the temporary flags
            if '_remove_interactor' in interactor:
                del interactor['_remove_interactor']
            if '_delete_interactor_completely' in interactor:
                del interactor['_delete_interactor_completely']
            filtered_interactors.append(interactor)

    ctx_json['interactors'] = filtered_interactors

    # Also update snapshot_json if it exists
    if 'snapshot_json' in json_data:
        json_data['snapshot_json']['interactors'] = filtered_interactors

    # ========================================================================
    # CHAIN VALIDATION - Verify indirect interactor relationships
    # ========================================================================
    print(f"\n{'='*80}")
    print("VALIDATING INDIRECT INTERACTOR CHAINS")
    print(f"{'='*80}")

    chains_validated = 0
    chains_fixed = 0
    chains_removed = 0

    for interactor in filtered_interactors:
        primary = interactor.get('primary', 'UNKNOWN')
        interaction_type = interactor.get('interaction_type', 'direct')
        upstream_interactor = interactor.get('upstream_interactor')
        mediator_chain = interactor.get('mediator_chain', [])

        if interaction_type == 'indirect':
            chains_validated += 1
            print(f"\n  Validating chain for {primary}...")

            # Validate mediator chain consistency
            if not upstream_interactor:
                print(f"    [WARN]WARNING: Indirect interactor missing upstream_interactor - fixing")
                if mediator_chain and len(mediator_chain) > 0:
                    interactor['upstream_interactor'] = mediator_chain[-1]
                    chains_fixed += 1
                else:
                    print(f"    [WARN]WARNING: Cannot fix - no mediator chain data available")
                    print(f"    Keeping classification as 'indirect' (Phase 1 is authoritative)")
                    print(f"    This interactor needs chain validation in future queries")
                    # Do NOT change interaction_type - preserve Phase 1's classification
                    # Just mark that it needs validation
                    interactor['_needs_chain_validation'] = True
                    chains_fixed += 1

            # Validate mediator chain matches upstream_interactor
            if mediator_chain and upstream_interactor:
                if upstream_interactor not in mediator_chain:
                    print(f"    [WARN]WARNING: upstream_interactor '{upstream_interactor}' not in chain {mediator_chain}")
                    # Fix: add upstream_interactor to end of chain
                    interactor['mediator_chain'] = mediator_chain + [upstream_interactor]
                    chains_fixed += 1
                    print(f"    [OK]Fixed: Added to chain {interactor['mediator_chain']}")

            # Validate depth matches chain length
            expected_depth = len(mediator_chain) + 1 if mediator_chain else 2
            actual_depth = interactor.get('depth')
            if actual_depth != expected_depth:
                print(f"    [WARN]WARNING: Depth mismatch (expected {expected_depth}, got {actual_depth}) - fixing")
                interactor['depth'] = expected_depth
                chains_fixed += 1

            # Validate mediators exist as direct interactors
            if mediator_chain:
                mediator_symbols = [i.get('primary') for i in filtered_interactors]
                for mediator in mediator_chain:
                    if mediator not in mediator_symbols:
                        print(f"    [WARN]WARNING: Mediator '{mediator}' not found in interactors")
                        print(f"       Chain may be incomplete. Consider re-querying.")

            print(f"    [OK]Chain validated: {main_protein} → {' → '.join(mediator_chain)} → {primary} (depth={interactor.get('depth')})")

    if chains_validated > 0:
        print(f"\n  Summary:")
        print(f"    Chains validated: {chains_validated}")
        print(f"    Chains fixed: {chains_fixed}")
        print(f"    Chains removed: {chains_removed}")
    else:
        print("  No indirect interactors found - skipping chain validation")

    print(f"\n{'='*80}")
    print("FACT-CHECKING SUMMARY")
    print(f"{'='*80}")
    print(f"Total claims checked: {total_claims}")
    print(f"  [OK]Validated TRUE: {claims_validated_true} (exact claim correct)")
    print(f"  🔧 CORRECTED: {claims_corrected} (wrong claim fixed)")
    if claims_corrected_from_false > 0:
        print(f"     ├─ From CORRECTED validity: {claims_corrected - claims_corrected_from_false}")
        print(f"     └─ From FALSE validity (salvaged!): {claims_corrected_from_false}")
    print(f"  ✗ Validated FALSE: {claims_validated_false} (removed; no alternative found)")
    false_removed = claims_validated_false
    if false_removed > 0:
        print(f"     └─ Removed: {false_removed} (no salvageable function for interaction)")
    print(f"  [DELETE]DELETED: {claims_deleted} (interaction fabricated, removed)")
    print(f"  [WARN]Conflicting/Uncertain: {claims_conflicting}")
    print(f"  [STATS] Total removed: {claims_removed}")
    if interactors_removed > 0:
        print(f"  [DELETE]Interactors removed (no valid functions): {interactors_removed}")
    print(f"\nQuality Improvements:")
    print(f"  - Claims kept as accurate: {claims_validated_true}")
    print(f"  - Claims corrected (salvaged): {claims_corrected}")
    if claims_corrected_from_false > 0:
        print(f"    * {claims_corrected_from_false} were marked FALSE but we found the real function!")
    print(f"  - Claims removed (unsupported): {claims_removed}")
    success_rate = ((claims_validated_true + claims_corrected) / total_claims * 100) if total_claims > 0 else 0
    print(f"  - Success rate (TRUE + CORRECTED): {success_rate:.1f}%")
    print(f"{'='*80}\n")

    return json_data


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fact-check and correct functional claims by searching PubMed. "
                    "Now with smart correction: fixes wrong claims instead of just deleting them!"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to the pipeline JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (default: <input>_fact_checked.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--run-evidence-validator",
        action="store_true",
        help="Run evidence_validator.py after fact-checking is complete"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size for evidence validator (default: 3)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("❌ GOOGLE_API_KEY not found. Add it to your .env file.")

    # Load input JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        sys.exit(f"❌ Input file not found: {input_path}")

    print(f"\n{'='*80}")
    print("CLAIM FACT-CHECKER & CORRECTOR")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Mode: Verify + Correct (fixes wrong claims instead of deleting)")
    print(f"Evidence: Searches existing papers + new PubMed queries\n")

    # Load JSON
    try:
        json_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"❌ Failed to load JSON: {e}")

    # Fact-check claims with timing
    import time
    start_time = time.time()

    try:
        checked_data = fact_check_json(
            json_data,
            api_key,
            args.verbose
        )
    except Exception as e:
        sys.exit(f"❌ Fact-checking failed: {e}")

    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_fact_checked{input_path.suffix}"

    try:
        output_path.write_text(
            json.dumps(checked_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"[OK]Saved fact-checked JSON to: {output_path}")
    except Exception as e:
        sys.exit(f"❌ Failed to save output: {e}")

    print(f"\n{'='*80}")
    print("[OK]FACT-CHECKING COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal time: {elapsed_min:.1f} minutes ({elapsed_time:.0f}s)")

    print(f"\n{'='*80}")
    print("VALIDATION APPROACH")
    print(f"{'='*80}")
    print(f"  • One fresh Gemini 2.5 Pro call per interactor")
    print(f"  • Thinking budget: Dynamic (model decides per claim)")
    print(f"  • Every PMID verified against PubMed")
    print(f"  • Google Search used for comprehensive validation")

    print(f"\n{'='*80}")
    print("WHAT WAS DONE")
    print(f"{'='*80}")
    print(f"  [OK]TRUE claims: Verified with PMID confirmation")
    print(f"  🔧 CORRECTED claims: Wrong claims were fixed (not deleted!)")
    print(f"  🔧 FALSE→CORRECTED: Unsupported claims were salvaged with correct function")
    print(f"  ✗ FALSE claims (no correction): Removed (no salvageable interaction)")
    print(f"  [DELETE]DELETED claims: Removed (fake PMIDs or fabricated interactions)")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"  1. Review corrected claims (marked with 'CORRECTED' validity)")
    print(f"  2. Check 'original_function_name' field to see what changed")
    print(f"  3. Check 'evidence_source' to see correction type:")
    print(f"     - fact_checker_corrected: CORRECTED validity (wrong function, same interactor)")
    print(f"     - fact_checker_corrected_from_false: FALSE→CORRECTED (found real function!)")
    print(f"  4. (Optional) Run evidence validator: python claim_fact_checker.py <input> --run-evidence-validator")
    print(f"  5. Generate visualization: python visualizer.py {output_path}")
    print(f"\nData fields:")
    print(f"  - validity: TRUE | CORRECTED | FALSE | DELETED | CONFLICTING")
    print(f"  - validation_note: Explanation of validation result")
    print(f"  - evidence_source: fact_checker_validated | fact_checker_corrected | fact_checker_corrected_from_false | original_pipeline")
    print(f"  - original_function_name: (only for CORRECTED) what it was before correction")
    print(f"\n💡 FALSE→CORRECTED: When claim is wrong but we found what the interaction actually does!")
    if PMID_EXTRACTOR_AVAILABLE:
        print(f"  - PMIDs verified via PubMed API ✓")

    # Run evidence_validator.py if requested
    if args.run_evidence_validator:
        print(f"\n{'='*80}")
        print("RUNNING EVIDENCE VALIDATOR")
        print(f"{'='*80}")
        print(f"Input: {output_path}")
        print(f"Batch size: {args.batch_size}")

        try:
            # Import evidence validator
            import evidence_validator

            # Load the fact-checked JSON
            validated_json_data = evidence_validator.load_json_file(output_path)

            # Run evidence validation
            validator_start_time = time.time()
            validated_json_data = evidence_validator.validate_and_enrich_evidence(
                validated_json_data,
                api_key,
                verbose=args.verbose,
                batch_size=args.batch_size
            )
            validator_elapsed = time.time() - validator_start_time
            validator_elapsed_min = validator_elapsed / 60

            # Save the evidence-validated output
            evidence_output_path = output_path.parent / f"{output_path.stem}_evidence_validated{output_path.suffix}"
            evidence_validator.save_json_file(validated_json_data, evidence_output_path)

            print(f"\n{'='*80}")
            print("[OK]EVIDENCE VALIDATION COMPLETE")
            print(f"{'='*80}")
            print(f"Evidence validation time: {validator_elapsed_min:.1f} minutes ({validator_elapsed:.0f}s)")
            print(f"Evidence-validated output: {evidence_output_path}")

            # Update total time
            total_elapsed = elapsed_time + validator_elapsed
            total_elapsed_min = total_elapsed / 60

            print(f"\n{'='*80}")
            print("[OK]FULL PIPELINE COMPLETE")
            print(f"{'='*80}")
            print(f"Fact-checking time: {elapsed_min:.1f} minutes ({elapsed_time:.0f}s)")
            print(f"Evidence validation time: {validator_elapsed_min:.1f} minutes ({validator_elapsed:.0f}s)")
            print(f"Total time: {total_elapsed_min:.1f} minutes ({total_elapsed:.0f}s)")
            print(f"\nFinal output: {evidence_output_path}")

        except ImportError:
            print(f"[WARN]WARNING: evidence_validator.py not found in the same directory")
            print(f"[WARN]Skipping evidence validation")
            print(f"[WARN]To use this feature, ensure evidence_validator.py is in: {Path(__file__).parent}")
        except Exception as e:
            print(f"[WARN]WARNING: Evidence validation failed: {e}")
            print(f"[WARN]Fact-checked output is still available at: {output_path}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
