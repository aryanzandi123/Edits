"""
Dynamic Pipeline Configuration Generator
Allows user to specify number of interactor and function discovery rounds
"""
from __future__ import annotations
from pipeline import config_gemini_MAXIMIZED as base_config
from pipeline.types import StepConfig


# Import all the base components we need
DIFFERENTIAL_OUTPUT_RULES = base_config.DIFFERENTIAL_OUTPUT_RULES
STRICT_GUARDRAILS = base_config.STRICT_GUARDRAILS
INTERACTOR_TYPES = base_config.INTERACTOR_TYPES  # NEW: replaces ARROW_TYPE_DECISION_TREE
FUNCTION_NAMING_RULES = base_config.FUNCTION_NAMING_RULES
SCHEMA_HELP = base_config.SCHEMA_HELP
MAX_OUTPUT_TOKENS = base_config.MAX_OUTPUT_TOKENS
DYNAMIC_SEARCH_THRESHOLD = base_config.DYNAMIC_SEARCH_THRESHOLD


def create_interactor_discovery_step(round_num: int) -> StepConfig:
    """
    Create an additional interactor discovery step dynamically.
    NEW: NAMES ONLY - NO arrows/directions/evidence yet!

    Args:
        round_num: Round number (4, 5, 6, etc.)

    Returns:
        StepConfig for this round
    """
    # Map numbers to letters: 4->f, 5->g, 6->h, etc.
    letter = chr(ord('f') + (round_num - 4))

    ordinals = {
        4: "Fourth", 5: "Fifth", 6: "Sixth", 7: "Seventh",
        8: "Eighth", 9: "Ninth", 10: "Tenth"
    }
    ordinal = ordinals.get(round_num, f"{round_num}th")

    return StepConfig(
        name=f"step1{letter}_discover_round{round_num}",
        model="gemini-2.5-pro",
        deep_research=False,
        reasoning_effort="high",
        use_google_search=True,
        thinking_budget=None,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        search_dynamic_mode=True,
        search_dynamic_threshold=DYNAMIC_SEARCH_THRESHOLD,
        expected_columns=["ctx_json", "step_json"],
        system_prompt=None,
        prompt_template=(
            DIFFERENTIAL_OUTPUT_RULES
            + "\n\n"
            + STRICT_GUARDRAILS
            + "\n\n"
            + INTERACTOR_TYPES
            + "\n\n"
            + "\n".join([
                "╔═══════════════════════════════════════════════════════════════╗",
                f"║  STEP 1{letter.upper()}: {ordinal.upper()} ROUND INTERACTOR DISCOVERY (NAMES)" + " " * (4 - len(ordinal)) + "║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "EXISTING: {ctx_json.interactor_history}",
                "",
                f"OBJECTIVE: Round {round_num} - Find 5-10 MORE protein names (direct/indirect)",
                "",
                "THIS STEP ONLY FINDS PROTEIN NAMES - DO NOT determine arrows/directions yet!",
                "Arrows/directions will be determined AFTER function discovery (Step 2c).",
                "",
                "SEARCH FREELY:",
                "Use your biological expertise to search creatively - NO rigid templates!",
                "Explore different angles based on what you discover.",
                "",
                "Examples (but don't limit yourself!):",
                "- '{main} protein interactions'",
                "- '{main} pathway'",
                "- '{main} regulates'",
                "- Whatever makes sense based on your research!",
                "",
                "FOR EACH PROTEIN FOUND:",
                "1. Classify as DIRECT or INDIRECT:",
                "   - DIRECT: Physical interaction (Co-IP, Y2H, BioID evidence)",
                "   - INDIRECT: Pathway/cascade partner (no direct binding)",
                "",
                "2. If INDIRECT, track the FULL CHAIN:",
                "   Example: '{main} activates VCP which activates LAMP2 which activates LAMP1'",
                "   → VCP = DIRECT (upstream_interactor = null, mediator_chain = [])",
                "   → LAMP2 = INDIRECT (upstream_interactor = 'VCP', mediator_chain = ['VCP'])",
                "   → LAMP1 = INDIRECT (upstream_interactor = 'LAMP2', mediator_chain = ['VCP', 'LAMP2'])",
                "",
                "3. Calculate depth: 1=direct, 2=first indirect, 3=second indirect, etc.",
                "",
                "4. Brief support_summary only (NO arrows/directions/evidence yet!)",
                "",
                "MINIMAL OUTPUT (with chain tracking!):",
                "{",
                "  'primary': '<PROTEIN>',",
                "  'interaction_type': 'direct' | 'indirect',",
                "  'upstream_interactor': null | '<PROTEIN>',",
                "  'mediator_chain': [] | ['PROTEIN1', 'PROTEIN2', ...],",
                "  'depth': 1 | 2 | 3 | ...,",
                "  'support_summary': '<brief context>'",
                "}",
                "",
                "DO NOT INCLUDE: arrow, direction, intent, evidence, paper_title, pmids, functions",
                "",
                "GOAL: Aggregate 5-10 NEW protein names for deep research later.",
                "",
                "Search freely! Use your AI reasoning - no constraints!",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    )


def create_function_mapping_step(round_num: int) -> StepConfig:
    """
    Create an additional function mapping step dynamically.
    NEW: Includes paper title collection + indirect interactor tracking

    Args:
        round_num: Round number (4, 5, 6, etc.)

    Returns:
        StepConfig for this round
    """
    # Function rounds: 2a4, 2a5, 2a6, etc.
    step_name = f"step2a{round_num}_functions_round{round_num}"

    ordinals = {
        4: "Fourth", 5: "Fifth", 6: "Sixth", 7: "Seventh",
        8: "Eighth", 9: "Ninth", 10: "Tenth"
    }
    ordinal = ordinals.get(round_num, f"{round_num}th")

    return StepConfig(
        name=step_name,
        model="gemini-2.5-pro",
        deep_research=False,
        reasoning_effort="high",
        use_google_search=True,
        thinking_budget=None,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        search_dynamic_mode=True,
        search_dynamic_threshold=DYNAMIC_SEARCH_THRESHOLD,
        expected_columns=["ctx_json", "step_json"],
        system_prompt=None,
        prompt_template=(
            DIFFERENTIAL_OUTPUT_RULES
            + "\n\n"
            + STRICT_GUARDRAILS
            + "\n\n"
            + FUNCTION_NAMING_RULES
            + "\n\n"
            + "\n".join([
                "╔═══════════════════════════════════════════════════════════════╗",
                f"║  STEP 2A{round_num}: {ordinal.upper()} ROUND FUNCTION + TITLES + INDIRECT" + " " * (1 - len(str(round_num))) + "║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "MAIN: {ctx_json.main}",
                "INTERACTORS: {ctx_json.interactor_history}",
                "COVERAGE: {ctx_json.function_batches}",
                "",
                f"TASK: Round {round_num} - Find MORE functions + COLLECT PAPER TITLES + TRACK INDIRECT INTERACTORS",
                "",
                "RESEARCH FREELY WITH YOUR BIOLOGICAL EXPERTISE:",
                "- Use Google Search creatively (NO rigid templates!)",
                "- Explore different angles based on what you discover",
                "- Follow biological leads in the literature",
                "",
                "FOR EACH FUNCTION FOUND:",
                "1. ULTRA-SPECIFIC function naming (see rules above)",
                "   - **UNIQUENESS:** Ensure functions are specific to the interaction pair! Do not copy-paste from other interactors.",
                "2. Set arrow: 'activates' or 'inhibits'",
                "3. Write effect_description (one sentence)",
                "4. Multiple biological_consequence cascades (WITH arrows →)",
                "5. Direct specific_effects (NO arrows)",
                "6. **COLLECT PAPER TITLES** (2-4 per function):",
                "   - NO title constraints!",
                "   - Collect ANY relevant paper about this function",
                "   - Title does NOT need query protein name",
                "   - Store: {'paper_title': '<FULL TITLE>', 'year': <int>}",
                "",
                "7. **TRACK INDIRECT INTERACTORS** in cascades with FULL CHAINS:",
                "   - If cascades mention OTHER proteins, add to indirect_interactors:",
                "     {",
                "       'name': '<PROTEIN>',",
                "       'upstream_interactor': '<PROTEIN>',",
                "       'mediator_chain': ['<MEDIATOR1>', '<MEDIATOR2>', ...],",
                "       'depth': <int>,",
                "       'discovered_in_function': '<function>',",
                "       'role_in_cascade': '<desc>'",
                "     }",
                "   - Example: '{main} → VCP → LAMP2 → LAMP1' in autophagy cascade:",
                "     → LAMP2: upstream_interactor='VCP', mediator_chain=['VCP'], depth=2",
                "     → LAMP1: upstream_interactor='LAMP2', mediator_chain=['VCP','LAMP2'], depth=3",
                "",
                "GOAL: Add 10-20 NEW, UNIQUE functions across interactors. Avoid duplicates.",
                "",
                "Search freely! No constraints on strategy!",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    )




def generate_pipeline(
    num_interactor_rounds: int = 3,
    num_function_rounds: int = 3,
    max_depth: int = 3
) -> list[StepConfig]:
    """
    Generate a complete pipeline with specified number of discovery rounds.

    NEW PIPELINE STRUCTURE:
    1. Interactor Discovery (1a-1g+) - Find protein names only (direct/indirect)
    2. Function Discovery (2a-2a5+, 2b) - Find functions + paper titles + indirect interactors
    3. Arrow Determination (2c) - GENERATED DYNAMICALLY BY RUNNER.PY (per-interactor)
    4. Final QC (2g) - Quality control
    5. Snapshot (3) - Create final output

    Args:
        num_interactor_rounds: Total interactor discovery rounds (default 3, min 1, max 10)
        num_function_rounds: Total function mapping rounds (default 3, min 1, max 10)
        max_depth: Maximum chain depth for indirect interactors (default 3)
                   1 = direct only, 2 = direct + 1 level indirect, 3 = 2 levels, etc.
                   5+ = unlimited depth

    Returns:
        List of StepConfig objects for the complete pipeline

    NOTE: Arrow determination steps (2c) are NOT included in this list.
          They are generated dynamically by runner.py based on interactor_history.
          The max_depth parameter is stored in pipeline metadata for use by runner.
    """
    # Validate inputs
    num_interactor_rounds = max(1, min(10, num_interactor_rounds))
    num_function_rounds = max(1, min(10, num_function_rounds))
    # For max_depth: 5+ means unlimited
    max_depth = max(1, max_depth) if max_depth < 5 else 999

    steps = []
    base_steps = base_config.PIPELINE_STEPS

    # Find insertion points
    interactor_insertion_idx = None
    function_insertion_idx = None
    indirect_functions_idx = None
    rescue_direct_functions_idx = None
    arrow_template_idx = None
    qc_idx = None

    for idx, step in enumerate(base_steps):
        if step.name == "step2a_functions":
            interactor_insertion_idx = idx
        elif step.name == "step2b_deep_function_research":
            function_insertion_idx = idx
        elif step.name == "step2b2_indirect_functions":
            indirect_functions_idx = idx
        elif step.name == "step2b3_rescue_direct_functions":
            rescue_direct_functions_idx = idx
        elif step.name == "step2c_arrow_TEMPLATE":
            arrow_template_idx = idx
        elif step.name == "step2g_final_qc":
            qc_idx = idx

    # 1. Add interactor discovery steps (1a-1g+)
    # Base has 7 steps (1a-1g), can extend beyond
    interactor_steps_to_add = min(num_interactor_rounds, 7)

    current_idx = 0
    steps_added = 0
    while current_idx < interactor_insertion_idx and steps_added < interactor_steps_to_add:
        steps.append(base_steps[current_idx])
        current_idx += 1
        steps_added += 1

    # Extra interactor rounds if requested (beyond base 7)
    if num_interactor_rounds > 7:
        for extra_round in range(8, num_interactor_rounds + 1):
            steps.append(create_interactor_discovery_step(extra_round))

    # 2. Add function mapping steps (2a-2a5+)
    # Base has 5 steps (2a-2a5), can extend beyond
    current_idx = interactor_insertion_idx
    function_steps_to_add = min(num_function_rounds, 5)

    steps_added = 0
    while current_idx < function_insertion_idx and steps_added < function_steps_to_add:
        steps.append(base_steps[current_idx])
        current_idx += 1
        steps_added += 1

    # Extra function rounds if requested (beyond base 5)
    if num_function_rounds > 5:
        for extra_round in range(6, num_function_rounds + 1):
            steps.append(create_function_mapping_step(extra_round))

    # 3. Add Step 2b (deep function research)
    if function_insertion_idx is not None:
        steps.append(base_steps[function_insertion_idx])

    # 3b. Add Step 2b2 (indirect functions generation)
    if indirect_functions_idx is not None:
        steps.append(base_steps[indirect_functions_idx])

    # 3c. Add Step 2b3 (rescue direct interactors without functions)
    if rescue_direct_functions_idx is not None:
        steps.append(base_steps[rescue_direct_functions_idx])

    # 4. SKIP arrow template step (step2c_arrow_TEMPLATE)
    # Arrow determination steps are generated dynamically by runner.py
    # See: runner.py will call create_arrow_determination_step() for each interactor

    # 5. Add Step 2g (final QC) and Step 3 (snapshot)
    if qc_idx is not None:
        for idx in range(qc_idx, len(base_steps)):
            # Skip the arrow template
            if base_steps[idx].name != "step2c_arrow_TEMPLATE":
                steps.append(base_steps[idx])

    return steps


def get_default_pipeline() -> list[StepConfig]:
    """Get the default pipeline (same as base config)."""
    return base_config.PIPELINE_STEPS


# For backwards compatibility, export PIPELINE_STEPS
PIPELINE_STEPS = get_default_pipeline()
