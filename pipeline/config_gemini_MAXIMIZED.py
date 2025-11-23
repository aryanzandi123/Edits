"""MAXIMIZED pipeline configuration for Gemini 2.5 Pro - RESTRUCTURED
- AUTO thinking tokens (model decides per scenario)
- MAX output tokens (65K) at EVERY step
- INTERACTOR DISCOVERY: Find protein names only (direct/indirect) - NO arrows yet!
- FUNCTION DISCOVERY: Find mechanisms + paper titles + indirect interactors
- ARROW DETERMINATION: Analyze normal roles vs interaction effects (NEW PHASE)
- Multi-cascade biological consequences
- Anti-hallucination guardrails
- EXTENSIVE ITERATION: 7 interactor discovery + 6 function mapping + per-interactor arrow rounds
- 20+ total steps to ensure comprehensive coverage (nothing missed!)
"""
from __future__ import annotations

from pipeline.types import StepConfig

# MAXIMUM CAPABILITIES
# Note: thinking_budget removed - model now decides dynamically per step
MAX_OUTPUT_TOKENS = 65536
DYNAMIC_SEARCH_THRESHOLD = 0.0

# ------------------------
# SHARED TEXT BLOCKS
# ------------------------

DIFFERENTIAL_OUTPUT_RULES = """╔═══════════════════════════════════════════════════════════════╗
║  DIFFERENTIAL OUTPUT RULES (CRITICAL FOR TOKEN EFFICIENCY)   ║
╚═══════════════════════════════════════════════════════════════╝
YOU MUST OUTPUT ONLY **NEW OR MODIFIED DATA** FROM THIS STEP!

DO NOT re-output existing interactors or functions from previous steps.
The runner will merge your incremental changes into the full context.

OUTPUT FORMAT:
{
  "ctx_json": {
    "main": "{user_query}",  // Always include this
    "interactors": [
      // ONLY interactors you are ADDING or MODIFYING in THIS step
      // If you're adding functions to existing interactor, include only that interactor with new functions
      // If you're adding new interactors, include only the new ones
    ],
    "interactor_history": ["NEW1", "NEW2", ...],  // ONLY newly discovered names
    "function_history": {"PROTEIN": [...]},       // ONLY updated entries
    "function_batches": ["BATCH1", ...],          // ONLY new batches processed
    "search_history": ["query1", ...]             // ONLY new search queries
  },
  "step_json": {"step": "...", ...}
}

EXAMPLES:
❌ BAD (re-outputting everything):
  "interactors": [all 32 interactors with all their functions]

✅ GOOD (incremental update):
  "interactors": [the 5 NEW interactors you just discovered]

❌ BAD (for function mapping step):
  "interactors": [all 32 interactors]

✅ GOOD (for function mapping step):
  "interactors": [
    {
      "primary": "PROTEIN_X",  // only this interactor
      "functions": [new function 1, new function 2]  // only NEW functions for this protein
    }
  ]

The runner will intelligently merge your output with existing data."""

STRICT_GUARDRAILS = """╔═══════════════════════════════════════════════════════════════╗
║  STRICT NON-HALLUCINATION RULES (ABSOLUTE OVERRIDE)         ║
╚═══════════════════════════════════════════════════════════════╝
- NEVER guess, infer, or fabricate ANY information
- Add claims ONLY when supported by verifiable primary evidence
- Evidence validation happens in claim_fact_checker.py - use placeholder evidence structure for now
- Output ONLY valid JSON (no markdown, no prose, no explanations)
- If uncertain about ANYTHING, search again or omit the claim"""

SCHEMA_HELP = """╔═══════════════════════════════════════════════════════════════╗
║  SCHEMA SPECIFICATION                                         ║
╚═══════════════════════════════════════════════════════════════╝

ctx_json = {
  'main': '<HGNC_SYMBOL>',
  'interactors': [{
      'primary': '<HGNC_SYMBOL>',

      // NEW: INTERACTION TYPE (determined in discovery phase)
      'interaction_type': 'direct' | 'indirect',
      'upstream_interactor': '<HGNC>' | null,  // null for direct, protein name for indirect
      'mediator_chain': ['<PROTEIN1>', '<PROTEIN2>', ...],  // Path from main to this interactor (empty for direct)
      'depth': 1 | 2 | 3,  // 1=direct, 2=one mediator, 3=two mediators

      // ARROW/DIRECTION (determined AFTER function discovery in Step 2b)
      'direction': 'main_to_primary' | 'primary_to_main' | 'bidirectional',
      'arrow': 'activates' | 'inhibits' | 'binds',
      'intent': 'phosphorylation' | 'ubiquitination' | 'binding' | ...,

      'multiple_mechanisms': true | false,
      'mechanism_details': ['<specific mechanism 1>', '<mechanism 2>', ...],

      'pmids': [],  // Extracted from paper titles in function discovery
      'evidence': [{
          'paper_title': '<FULL title from literature>',
          'relevant_quote': '<key quote from paper>',
          'year': <int>,
          'doi': '<DOI if available>',
          'authors': '<first author et al.>',
          'journal': '<journal name>',
          'assay': '<experimental method>',
          'species': '<organism>'
      }],

      'support_summary': '<=160 chars summary',

      // FUNCTIONS (discovered in Step 2a-2a5, enriched with paper titles)
      'functions': [{
          'function': '<short function name>',
          'arrow': 'activates' | 'inhibits',
          'cellular_process': '<detailed mechanism>',
          'effect_description': '<What happens to the function as a result>',

          'biological_consequence': [
              '<cascade 1: pathway → intermediate → outcome>',
              '<cascade 2: different pathway → different outcome>'
          ],

          'specific_effects': [
              '<Direct, concise outcome - NO ARROWS>',
              '<Another direct effect>'
          ],

          'pmids': [],
          'evidence': [{
              'paper_title': '<FULL title mentioning this function>',
              'relevant_quote': '<key mechanism quote>',
              'year': <int>
          }],
          'mechanism_id': '<link to mechanism_details>',
          'note': '<optional clarification>',
          'normal_role': '<optional canonical function context>'
      }, ...]
  }, ...],

  // TRACKING FIELDS
  'interactor_history': ['<HGNC>', ...],
  'indirect_interactors': [{
      'name': '<HGNC>',
      'upstream_interactor': '<HGNC>',
      'discovered_in_function': '<function name>',
      'role_in_cascade': '<description>',
      'depth': <int>  // 0=query, 1=direct, 2+=indirect
  }],
  'function_history': {'<PROTEIN>': [<functions>], ...},
  'function_batches': ['<HGNC>', ...],
  'search_history': ['<query 1>', '<query 2>', ...]
}

KEY REMINDERS (NEW PIPELINE STRUCTURE):
- Interactor discovery: Find protein NAMES only (direct/indirect)
- Function discovery: Find mechanisms + COLLECT PAPER TITLES for each function
- Arrow determination: Happens AFTER all functions discovered (Step 2b)
- Paper titles: NO constraints (collect ANY relevant title, even without query name)
- Indirect interactors: Track proteins discovered in cascades during function discovery
"""

FUNCTION_NAMING_RULES = """╔═══════════════════════════════════════════════════════════════╗
║  ULTRA-SPECIFIC FUNCTION NAMING (ABSOLUTE REQUIREMENT)       ║
╚═══════════════════════════════════════════════════════════════╝

The 'function' field is THE MOST CRITICAL part of each function entry.
It must PRECISELY name the ACTUAL molecular/cellular process being modulated.

When paired with the arrow (activates/inhibits), it must IMMEDIATELY answer:
"What SPECIFIC biological process/pathway/molecular activity is being affected?"

╔═══════════════════════════════════════════════════════════════╗
║  CRITICAL RULE #1: NEVER USE "REGULATION" IN FUNCTION NAMES  ║
╚═══════════════════════════════════════════════════════════════╝

The word "regulation" is BANNED from function names because the arrow
field already indicates the regulatory relationship!

✗✗✗ BANNED EXAMPLES (contain "regulation"):
  - "Apoptosis Regulation" → Use: "Apoptosis"
  - "Mitophagy Regulation" → Use: "Mitophagy"
  - "ATXN3 Aggregation Regulation" → Use: "ATXN3 Aggregation"
  - "Cell Cycle Regulation" → Use: "Cell Cycle Progression"
  - "Transcription Regulation" → Use: "Transcription"

WHY: The arrow field (activates/inhibits/promotes/suppresses) already tells us
HOW the function is being regulated. Adding "regulation" is redundant.

TEST YOUR FUNCTION NAME:
"The interaction [activates/inhibits] [FUNCTION NAME]"
  ✓ GOOD: "The interaction activates Apoptosis"
  ✗ BAD: "The interaction activates Apoptosis Regulation"

✓ EXCELLENT EXAMPLES (molecular processes are crystal clear, NO "regulation"):
  - "mTORC1 Kinase Activity & Protein Synthesis & Cell Growth"
  - "NF-κB Transcriptional Activity & Pro-Inflammatory Gene Expression"
  - "Caspase-3 Protease Activity & Apoptotic Cell Death"
  - "Autophagosome-Lysosome Fusion & Protein Degradation"
  - "CDK1/Cyclin B Complex Activity & G2/M Cell Cycle Progression"
  - "TP53 Protein Stability & DNA Damage Response Gene Expression"
  - "AMPK Kinase Activity & Cellular Energy Homeostasis"
  - "ROS Production & Mitochondrial Membrane Depolarization"
  - "Apoptosis" (NOT "Apoptosis Regulation")
  - "Mitophagy" (NOT "Mitophagy Regulation")
  - "ATXN3 Stability" (NOT "Regulation of ATXN3 Stability")
  - "ATXN3 Aggregation" (NOT "ATXN3 Aggregation Regulation")

✗ TERRIBLE EXAMPLES (too vague, meta-level, or generic):
  - "Regulation of Immunometabolism" → What is being regulated? HOW?
  - "Cell Survival" → Which pathway? What molecular mechanism?
  - "Protein Quality Control" → WAY too broad - which system?
  - "Metabolic Homeostasis" → Meaningless without molecular specifics
  - "Bacterial Killing" → What cellular process enables this?
  - "Immune Response" → Which pathway? What's being activated/inhibited?
  - "Cell Death" → Which form? What pathway?

GOLDEN RULE:
If someone reads ONLY the function name + arrow direction and can't tell
what molecular/cellular process is affected, THE NAME IS TOO VAGUE.

The function name should capture:
1. The specific molecular target or pathway (e.g., "mTORC1", "NF-κB", "TP53")
2. The type of activity/process (e.g., "Kinase Activity", "Transcription", "Stability")
3. Optionally, the downstream biological consequence (e.g., "Cell Growth", "Apoptosis")

WINNING FORMAT PATTERNS:
  "[Molecular Target] [Activity Type] & [Biological Outcome]"
  "[Pathway Name] Signaling & [Direct Result]"
  "[Protein Name] [Property] & [Downstream Effect]"
  "[Process Name]" (e.g., "Apoptosis", "Autophagy", "Mitophagy")

REMEMBER:
- Name the biological PROCESS, not its "regulation"
- The arrow field handles the regulatory relationship
- If you wrote "X Regulation", just write "X" instead

REAL-WORLD TEST:
Ask yourself: "If this function name appeared with an inhibitory arrow,
would a biologist immediately understand what cellular process is being blocked?"

If NO → The name is too vague. Make it MORE SPECIFIC.
If YES → Perfect! That's the level of precision required.

╔═══════════════════════════════════════════════════════════════╗
║  CRITICAL RULE #2: NEVER USE OUTCOME-BASED FUNCTION NAMES   ║
╚═══════════════════════════════════════════════════════════════╝

Function names must describe the TARGET CELLULAR PROCESS, NOT the OUTCOME.
The arrow field already indicates the EFFECT (activates/inhibits).

WHY THIS MATTERS: Outcome-based names create CONFUSING DOUBLE-NEGATIVES.

⚠️  DOUBLE-NEGATIVE DISASTER EXAMPLE:
  ✗ Function: "Apoptosis Suppression"
  ✗ Arrow: "inhibits"
  ✗ Read aloud: "Akt1 → VCP INHIBITS Apoptosis Suppression"
  ✗ Sounds like: Inhibiting suppression = PROMOTING apoptosis (WRONG!)

  ✓ Function: "Apoptosis"
  ✓ Arrow: "inhibits"
  ✓ Read aloud: "Akt1 → VCP INHIBITS Apoptosis"
  ✓ Correctly means: SUPPRESSING apoptosis (RIGHT!)

✗✗✗ BANNED OUTCOME-BASED PATTERNS:
  - "Apoptosis Suppression" → Use: "Apoptosis"
  - "Cell Survival Promotion" → Use: "Cell Survival" OR better: specify pathway
  - "Autophagy Induction" → Use: "Autophagy"
  - "Growth Inhibition" → Use: "Cell Growth"
  - "Transcription Activation" → Use: "Transcription"
  - "NF-κB Activation" → Use: "NF-κB Transcriptional Activity"
  - "mTOR Inhibition" → Use: "mTOR Kinase Activity"
  - "Protein Degradation Enhancement" → Use: "Protein Degradation"

BANNED VERB FORMS IN FUNCTION NAMES:
  - Suppression (use the base process: "Apoptosis", not "Apoptosis Suppression")
  - Activation (use the activity: "Kinase Activity", not "Kinase Activation")
  - Inhibition (use the process: "Growth", not "Growth Inhibition")
  - Promotion (use the outcome: "Cell Survival", not "Cell Survival Promotion")
  - Induction (use the process: "Autophagy", not "Autophagy Induction")
  - Stimulation (use the activity: "Transcription", not "Transcription Stimulation")
  - Enhancement (use the process: "Degradation", not "Degradation Enhancement")

TEST YOUR FUNCTION NAME:
1. Does it contain a verb form ending in -tion, -sion, or -ment?
   (Suppression, Activation, Inhibition, Promotion, Enhancement)
   → If YES, remove it and use the base process name

2. Would it create a double-negative when paired with "inhibits"?
   → If YES, simplify to just the target process

3. Can you read "inhibits [FUNCTION]" without confusion?
   → If NO, the name needs simplification

REAL-WORLD BIOLOGY TRANSLATION EXAMPLES:

Paper: "Akt suppresses apoptosis via VCP phosphorylation"
  ✗ BAD: Function="Apoptosis Suppression", Arrow="inhibits"
         (Double-negative: "inhibits suppression" confusing!)
  ✓ GOOD: Function="Apoptosis", Arrow="inhibits"
          (Clear: "inhibits apoptosis" = prevents cell death)

Paper: "VCP activates autophagy through mTOR inhibition"
  ✗ BAD: Function="Autophagy Activation", Arrow="activates"
         (Redundant: "activates activation")
  ✓ GOOD: Function="Autophagy", Arrow="activates"
          (Clear: "activates autophagy")

Paper: "mTOR inhibition promotes TFEB nuclear translocation"
  ✗ BAD: Function="TFEB Nuclear Translocation Promotion", Arrow="activates"
         (Redundant and wordy)
  ✓ GOOD: Function="TFEB Nuclear Translocation", Arrow="activates"
          (Clear and concise)

Paper: "HDAC6 enhances protein aggregate clearance"
  ✗ BAD: Function="Protein Aggregate Clearance Enhancement", Arrow="activates"
  ✓ GOOD: Function="Protein Aggregate Clearance", Arrow="activates"

WINNING PATTERN:
  - Name the TARGET PROCESS: "Apoptosis", "Autophagy", "Cell Cycle", "Transcription"
  - Let the ARROW indicate the EFFECT: "activates" or "inhibits"
  - Avoid verb forms: Remove -tion, -sion, -ment suffixes
  - Think: "What cellular process is being directly affected?"

REMEMBER:
- Function name = THE THING being affected (the target process)
- Arrow = HOW it's being affected (the direction of effect)
- Never use outcome verbs (Suppression, Activation, etc.) in function names
- The arrow already conveys activation/inhibition - don't duplicate it!

CLARITY TEST:
Read the function name + arrow aloud to a colleague.
If they get confused or have to re-read it → SIMPLIFY THE NAME.
If they immediately understand → Perfect!"""

INTERACTOR_TYPES = """╔═══════════════════════════════════════════════════════════════╗
║  INTERACTOR CLASSIFICATION (DIRECT vs INDIRECT)              ║
╚═══════════════════════════════════════════════════════════════╝

⚠️  CRITICAL: Classification is SET in Phase 1 and PRESERVED throughout pipeline!
You MUST classify EACH interactor based on EVIDENCE TYPE (not just description).

═══════════════════════════════════════════════════════════════════
DIRECT INTERACTORS (Physical/Molecular Binding)
═══════════════════════════════════════════════════════════════════

**Evidence types that indicate DIRECT interaction:**
✓ Co-immunoprecipitation (Co-IP)
✓ Pull-down assays (GST pull-down, FLAG-tag, etc.)
✓ Yeast two-hybrid (Y2H)
✓ BioID proximity labeling
✓ Förster resonance energy transfer (FRET)
✓ Surface plasmon resonance (SPR)
✓ Isothermal titration calorimetry (ITC)
✓ Crosslinking mass spectrometry (XL-MS)
✓ Structural data (X-ray crystallography, cryo-EM showing complex)
✓ Fluorescence polarization assays
✓ Native mass spectrometry
✓ Papers saying "binds to", "forms complex with", "interacts directly"

**How to classify:**
- If literature mentions ANY of the above assays → interaction_type: "direct"
- If papers explicitly say "direct interaction" → interaction_type: "direct"
- If structural data shows them in same complex → interaction_type: "direct"

═══════════════════════════════════════════════════════════════════
INDIRECT INTERACTORS (Functional/Pathway Relationship)
═══════════════════════════════════════════════════════════════════

**Evidence types that indicate INDIRECT interaction:**
✓ Genetic epistasis (double mutant analysis)
✓ Pathway analysis without binding evidence
✓ Multi-step cascades (A → B → C means A and C are indirect)
✓ Transcriptional regulation (unless TF-DNA binding shown directly)
✓ Functional assays (phosphorylation, ubiquitination) without binding evidence
✓ Papers saying "regulates", "modulates", "affects" WITHOUT binding data
✓ Downstream in signaling pathway
✓ Identified through RNA-seq, proteomics, but no binding shown

**How to classify:**
- If protein appears in CASCADE description → interaction_type: "indirect"
  - Example: "VCP activates mTOR which phosphorylates S6K"
  - S6K is INDIRECT (through mTOR)
- If only functional relationship shown → interaction_type: "indirect"
- If papers say "indirectly regulates" → interaction_type: "indirect"

**For INDIRECT interactors, you MUST set:**
- interaction_type: "indirect"
- upstream_interactor: "<protein that mediates the connection>" OR null

**Two types of indirect interactors:**

1. **Multi-hop indirect** (mediator known):
   - Example: VCP → mTOR → S6K means S6K has upstream_interactor: "mTOR"
   - Set: interaction_type="indirect", upstream_interactor="mTOR"

2. **First-ring indirect** (mediator unknown):
   - Example: VCP functionally regulates TFEB but direct mediator not elucidated
   - Evidence: Only functional assays ("VCP regulates TFEB nuclear translocation")
   - Set: interaction_type="indirect", upstream_interactor=null
   - This indicates indirect relationship exists but pathway is incomplete

═══════════════════════════════════════════════════════════════════
DECISION TREE FOR CLASSIFICATION
═══════════════════════════════════════════════════════════════════

For EACH interactor, ask:

1. **Does literature mention Co-IP, Y2H, BioID, or other binding assay?**
   → YES: interaction_type = "direct"
   → NO: Continue to question 2

2. **Does literature explicitly say "binds to" or "forms complex"?**
   → YES: interaction_type = "direct"
   → NO: Continue to question 3

3. **Is protein mentioned in multi-step cascade (A→B→C)?**
   → YES: interaction_type = "indirect", set upstream_interactor to mediator
   → NO: Continue to question 4

4. **Does literature only show functional relationship ("regulates", "affects")?**
   → YES: interaction_type = "indirect", upstream_interactor = null (first-ring indirect)
   → NO: Default to "direct" if clearly connected, omit if uncertain

═══════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════

Example 1: VCP and NPLOC4
Literature: "Co-IP shows VCP forms a complex with NPLOC4 and UFD1L"
→ Classification: interaction_type = "direct" (Co-IP evidence)

Example 2: VCP and S6K
Literature: "VCP activates mTORC1, which phosphorylates S6K"
→ Classification: interaction_type = "indirect", upstream_interactor = "mTORC1"
→ Reasoning: S6K is downstream in cascade, no direct binding evidence

Example 3: VCP and LAMP2
Literature: "VCP regulates autophagosome-lysosome fusion through LC3"
→ Classification: interaction_type = "indirect", upstream_interactor = "LC3"
→ Reasoning: LAMP2 affected through LC3, no VCP-LAMP2 binding shown

Example 4: p53 and MDM2
Literature: "Y2H screen identified MDM2 as p53-binding partner"
→ Classification: interaction_type = "direct" (Y2H evidence)

Example 5: VCP and TFEB (FIRST-RING INDIRECT)
Literature: "VCP regulates TFEB nuclear translocation in response to starvation"
→ Classification: interaction_type = "indirect", upstream_interactor = null
→ Reasoning: Only functional relationship shown ("regulates"), no binding evidence
→ Mediator unknown: VCP affects TFEB but direct mediator not elucidated in literature
→ This is first-ring indirect: indirect by nature, but no mediator specified

═══════════════════════════════════════════════════════════════════
VALIDATION CHECKLIST
═══════════════════════════════════════════════════════════════════

Before outputting, verify:
✓ EVERY interactor has interaction_type: "direct" or "indirect"
✓ INDIRECT interactors have upstream_interactor set to mediator OR null if unknown
✓ Classification matches evidence type (not just paper phrasing)
✓ Cascade proteins are marked indirect with proper upstream
"""

# ------------------------
# PIPELINE STEPS
# ------------------------

PIPELINE_STEPS: list[StepConfig] = [
    
    # 1a - Initial interactor discovery (NAMES ONLY - NO ARROWS YET!)
    StepConfig(
        name="step1a_discover",
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
                "║  STEP 1a: DISCOVER INTERACTOR PROTEINS (NAMES ONLY)         ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "QUERY PROTEIN: {user_query}",
                "",
                "CRITICAL: Initialize ctx_json with 'main': '{user_query}' as the first field!",
                "",
                "PRIMARY OBJECTIVE:",
                "Find 15-25 proteins that are RELEVANTLY mentioned with {user_query} in scientific literature.",
                "",
                "THIS STEP ONLY AGGREGATES PROTEIN NAMES - DO NOT determine arrows/directions yet!",
                "Those will be determined AFTER function discovery (in Step 2b).",
                "",
                "WHAT TO DO:",
                "1. Search freely using your biological expertise - NO rigid templates!",
                "   Examples: '{user_query} protein interactions'",
                "             '{user_query} binding partners'",
                "             '{user_query} pathway'",
                "             '{user_query} regulates'",
                "   Use whatever search strategy makes sense based on what you discover!",
                "",
                "2. For EACH protein found, classify as DIRECT or INDIRECT:",
                "   - DIRECT: Physical interaction (Co-IP, pull-down, Y2H, BioID evidence)",
                "   - INDIRECT: Same pathway/cascade, but no direct binding",
                "",
                "3. For INDIRECT interactors, you MUST provide chain data:",
                "   ",
                "   CRITICAL FIELDS (all required for indirect interactors):",
                "   - upstream_interactor: The protein that DIRECTLY interacts with this indirect interactor",
                "   - mediator_chain: Array of proteins forming the path from {user_query} to this interactor",
                "   - depth: Number of hops (1=direct, 2=one mediator, 3=two mediators)",
                "   ",
                "   Example 1 (one mediator):",
                "   If '{user_query} activates VCP, and VCP then activates LAMP2'",
                "   → VCP is DIRECT:",
                "      interaction_type: 'direct'",
                "      upstream_interactor: null",
                "      mediator_chain: []",
                "      depth: 1",
                "   → LAMP2 is INDIRECT (via VCP):",
                "      interaction_type: 'indirect'",
                "      upstream_interactor: 'VCP'  // LAMP2's direct partner",
                "      mediator_chain: ['VCP']      // Path: {user_query} → VCP → LAMP2",
                "      depth: 2                      // 2 hops",
                "   ",
                "   Example 2 (two mediators):",
                "   If '{user_query} activates VCP → VCP activates LAMP2 → LAMP2 activates Catalase'",
                "   → Catalase is INDIRECT (via VCP and LAMP2):",
                "      interaction_type: 'indirect'",
                "      upstream_interactor: 'LAMP2'      // Catalase's direct partner",
                "      mediator_chain: ['VCP', 'LAMP2']  // Full path",
                "      depth: 3                           // 3 hops",
                "",
                "MINIMAL OUTPUT STRUCTURE (NO arrows/directions/evidence yet!):",
                "{",
                "  'ctx_json': {",
                "    'main': '{user_query}',",
                "    'interactors': [",
                "      // DIRECT interactor example:",
                "      {",
                "        'primary': 'VCP',",
                "        'interaction_type': 'direct',",
                "        'upstream_interactor': null,",
                "        'mediator_chain': [],",
                "        'depth': 1,",
                "        'support_summary': 'VCP physically binds {user_query} (Co-IP evidence)'",
                "      },",
                "      // INDIRECT interactor example:",
                "      {",
                "        'primary': 'LAMP2',",
                "        'interaction_type': 'indirect',",
                "        'upstream_interactor': 'VCP',  // REQUIRED for indirect",
                "        'mediator_chain': ['VCP'],      // REQUIRED for indirect",
                "        'depth': 2,                     // REQUIRED for indirect",
                "        'support_summary': '{user_query} regulates LAMP2 via VCP'",
                "      }",
                "    ],",
                "    'interactor_history': ['VCP', 'LAMP2', ...],",
                "    'search_history': ['query1', 'query2', ...]",
                "  }",
                "}",
                "",
                "DO NOT INCLUDE:",
                "- arrow (determined later in Step 2b)",
                "- direction (determined later in Step 2b)",
                "- intent (determined later in Step 2b)",
                "- evidence array (collected in function discovery)",
                "- paper titles (collected in function discovery)",
                "- pmids (collected in function discovery)",
                "- functions (mapped in Step 2a-2a5)",
                "",
                "REMEMBER:",
                "- This step is about FINDING protein names to research later",
                "- Search freely based on your AI reasoning (no constraints!)",
                "- Mark each as direct or indirect",
                "- The deep mechanism analysis happens in function discovery",
            ])
            + "\n\n"
            + "Return ONLY JSON with ctx_json and step_json={'step':'step1a_discover','count':<n>}"
        ),
    ),

    # 1b - Expand interactor network (NAMES ONLY)
    StepConfig(
        name="step1b_expand",
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
                "║  STEP 1b: EXPAND INTERACTOR NETWORK (NAMES ONLY)            ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "CURRENT PROTEIN: {ctx_json.main}",
                "EXISTING INTERACTORS: {ctx_json.interactor_history}",
                "",
                "OBJECTIVE: Add 10-15 MORE protein names (direct/indirect)",
                "",
                "Search freely based on your biological reasoning - NO templates!",
                "Explore different angles that make sense for this protein.",
                "",
                "For each NEW protein:",
                "- Mark as 'direct' or 'indirect'",
                "- If INDIRECT, MUST include: upstream_interactor, mediator_chain, depth",
                "  (See Step 1a for detailed examples of chain data structure)",
                "- Brief support_summary only (NO arrows/directions/evidence yet)",
                "",
                "Remember: This step aggregates protein names for deep research later.",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 1c - Deep literature mining (NAMES ONLY)
    StepConfig(
        name="step1c_deep_mining",
        model="gemini-2.5-pro",
        deep_research=True,
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
                "║  STEP 1c: DEEP LITERATURE MINING (NAMES ONLY)               ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "EXISTING: {ctx_json.interactor_history}",
                "",
                "Use deep research to find 8-12 MORE protein names.",
                "Search freely - use your biological expertise to explore creatively.",
                "",
                "Mark each as direct/indirect.",
                "For INDIRECT proteins, include: upstream_interactor, mediator_chain, depth",
                "(See Step 1a for detailed chain data structure and examples)",
                "NO arrows/directions/evidence - just aggregating names for later research.",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 1d-1g - Additional interactor discovery rounds (NAMES ONLY)
    StepConfig(
        name="step1d_discover_round2",
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
                "║  STEP 1d: ROUND 2 INTERACTOR DISCOVERY (NAMES ONLY)         ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "EXISTING: {ctx_json.interactor_history}",
                "",
                "Add 5-10 MORE protein names. Search freely with no constraints.",
                "Mark as direct/indirect.",
                "For INDIRECT: include upstream_interactor, mediator_chain, depth (see Step 1a).",
                "NO arrows/directions/evidence yet.",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    StepConfig(
        name="step1e_discover_round3",
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
                "║  STEP 1e: ROUND 3 INTERACTOR DISCOVERY (NAMES ONLY)         ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "EXISTING: {ctx_json.interactor_history}",
                "",
                "Add 5-8 MORE protein names. Search creatively.",
                "Mark as direct/indirect.",
                "For INDIRECT: include upstream_interactor, mediator_chain, depth (see Step 1a).",
                "NO arrows/directions/evidence yet.",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    StepConfig(
        name="step1f_discover_round4",
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
                "║  STEP 1f: ROUND 4 INTERACTOR DISCOVERY (NAMES ONLY)         ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "EXISTING: {ctx_json.interactor_history}",
                "",
                "Add 3-6 MORE protein names. Explore new angles.",
                "Mark as direct/indirect.",
                "For INDIRECT: include upstream_interactor, mediator_chain, depth (see Step 1a).",
                "NO arrows/directions/evidence yet.",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    StepConfig(
        name="step1g_discover_round5",
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
                "║  STEP 1g: ROUND 5 INTERACTOR DISCOVERY (NAMES ONLY)         ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "EXISTING: {ctx_json.interactor_history}",
                "",
                "Final sweep - add 3-5 MORE protein names to ensure comprehensive coverage.",
                "Mark as direct/indirect.",
                "For INDIRECT: include upstream_interactor, mediator_chain, depth (see Step 1a).",
                "NO arrows/directions/evidence yet.",
            ])
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2a - MAP FUNCTIONS (first batch) + TRACK INDIRECT INTERACTORS
    StepConfig(
        name="step2a_functions",
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
                "║  STEP 2a: MAP FUNCTIONS + COLLECT PAPER TITLES + TRACK       ║",
                "║           INDIRECT INTERACTORS                                ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "MAIN: {ctx_json.main}",
                "INTERACTORS: {ctx_json.interactor_history}",
                "ALREADY PROCESSED: {ctx_json.function_batches}",
                "",
                "TASK: Process UNPROCESSED interactors (find ALL functions + paper titles)",
                "",
                "STEP 1: IDENTIFY TARGET INTERACTORS",
                "- Look at interactor_history (all discovered proteins)",
                "- Look at function_batches (proteins already processed)",
                "- Target = interactors NOT in function_batches",
                "- Process the FIRST 8-10 unprocessed interactors",
                "",
                "STEP 2: CREATE EXPLICIT LIST",
                "Before researching, write out your target list:",
                "  'Target interactors for this batch: [NAME1, NAME2, NAME3, ...]'",
                "",
                "STEP 3: PROCESS EACH INTERACTOR",
                "FOR EACH INTERACTOR:",
                "1. Search freely with NO constraints - use your biological expertise!",
                "2. Find ALL functions affected by query-interactor interaction",
                "3. For EACH function found:",
                "   a) **Name with ULTRA-PRECISION** (see rules above)",
                "   b) Set arrow (FUNCTION effect): 'activates' or 'inhibits'  # Effect on THIS FUNCTION",
                "   c) Describe cellular_process (HOW it works)",
                "   d) Write effect_description (ONE SENTENCE)",
                "   e) List biological_consequence cascades (MULTIPLE, with arrows →)",
                "   f) List specific_effects (direct facts, NO arrows)",
                "   g) **COLLECT PAPER TITLES** for this function:",
                "      - Find 2-4 papers discussing THIS SPECIFIC FUNCTION",
                "      - NO title constraints! Collect ANY relevant paper",
                "      - Title does NOT need to contain query protein name",
                "      - Just needs to discuss the function/mechanism",
                "      - Store in evidence array: {'paper_title': '<FULL TITLE>', 'year': <int>}",
                "",
                "4. **TRACK INDIRECT INTERACTORS** in cascades:",
                "   - If cascades mention OTHER proteins (e.g., 'VCP activates mTOR which activates S6K')",
                "   - Add them to indirect_interactors array:",
                "     {",
                "       'name': 'S6K',",
                "       'upstream_interactor': 'mTOR',",
                "       'discovered_in_function': '<function name>',",
                "       'role_in_cascade': 'downstream effector'",
                "     }",
                "",
                "PAPER TITLE EXAMPLE:",
                "Function: 'mTORC1 Kinase Activity & Protein Synthesis'",
                "Paper titles collected:",
                "  - 'VCP regulates mTORC1 signaling through amino acid sensing'",
                "  - 'Mechanisms of mTOR activation in cancer cells'",
                "  - 'Role of VCP in nutrient-dependent mTOR regulation'",
                "",
                "NOTE: Titles don't ALL need query name - just need to discuss the mechanism!",
                "",
                "STEP 4: MANDATORY OUTPUT TRACKING",
                "You MUST include in your output:",
                "  'function_batches': [ALL interactors you processed in this batch]",
                "",
                "EXAMPLE OUTPUT:",
                "{",
                "  'ctx_json': {",
                "    'main': 'VCP',",
                "    'interactors': [",
                "      {'primary': 'UFD1L', 'functions': [...]},",
                "      {'primary': 'NPLOC4', 'functions': [...]},",
                "      ... (all 8-10 processed interactors)",
                "    ],",
                "    'function_batches': ['UFD1L', 'NPLOC4', ...],  // ⚠️ CRITICAL: List ALL processed!",
                "    'indirect_interactors': [...]",
                "  }",
                "}",
                "",
                "⚠️  If you processed 8 interactors, function_batches MUST have 8 names!",
                "",
                "OUTPUT STRUCTURE:",
                "- functions array with paper_title in each evidence entry",
                "- indirect_interactors array for cascade proteins discovered",
                "- function_batches array with ALL processed interactor names (MANDATORY!)",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2a2 - Continue function mapping + paper titles + indirect tracking
    StepConfig(
        name="step2a2_functions_batch",
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
                "║  STEP 2a2: CONTINUE FUNCTION MAPPING + TITLES + INDIRECT     ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "ALREADY PROCESSED: {ctx_json.function_batches}",
                "",
                "TASK: Process NEXT batch of unprocessed interactors",
                "",
                "1. Identify interactors NOT in function_batches",
                "2. Process next 8-10 unprocessed interactors",
                "3. Generate functions + collect paper titles",
                "4. Track indirect interactors in cascades",
                "5. **MANDATORY**: Add ALL processed names to function_batches output",
                "",
                "Same requirements as Step 2a:",
                "- ULTRA-SPECIFIC function naming",
                "- COLLECT PAPER TITLES for EACH function (NO constraints on titles!)",
                "- TRACK INDIRECT INTERACTORS found in cascades",
                "- Search freely with your biological expertise",
                "",
                "⚠️  function_batches output MUST list ALL interactors processed in this batch!",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2a3 - Exhaustive function sweep + paper titles + indirect tracking
    StepConfig(
        name="step2a3_functions_exhaustive",
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
                "║  STEP 2a3: EXHAUSTIVE FUNCTION SWEEP + TITLES + INDIRECT     ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "ALREADY PROCESSED: {ctx_json.function_batches}",
                "",
                "TASK: Process ALL REMAINING unprocessed interactors (100% coverage)",
                "",
                "1. Identify ALL interactors NOT yet in function_batches",
                "2. Process EVERY single one of them (no skipping!)",
                "3. Generate functions + collect paper titles for each",
                "4. Track indirect interactors in cascades",
                "5. **MANDATORY**: Add ALL processed names to function_batches output",
                "",
                "Requirements:",
                "- ULTRA-SPECIFIC function naming",
                "- COLLECT PAPER TITLES for each function (ANY relevant paper!)",
                "- TRACK INDIRECT INTERACTORS from cascades",
                "- NO interactor should be left without functions after this step",
                "",
                "⚠️  VERIFICATION: After processing, function_batches should equal interactor_history!",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2a4 - Round 2 function mapping + paper titles + indirect tracking
    StepConfig(
        name="step2a4_functions_round2",
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
                "║  STEP 2a4: ROUND 2 FUNCTION MAPPING + TITLES + INDIRECT      ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "REVISIT interactors for ADDITIONAL functions:",
                "- Alternative outcomes, context-dependent functions",
                "- Recent papers (2020-2025)",
                "- COLLECT PAPER TITLES (no constraints!)",
                "- TRACK INDIRECT INTERACTORS from new cascades",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2a5 - Round 3 function mapping + paper titles + indirect tracking
    StepConfig(
        name="step2a5_functions_round3",
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
                "║  STEP 2a5: ROUND 3 FUNCTION MAPPING + TITLES + INDIRECT      ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "FINAL function sweep - use your biological expertise freely!",
                "Search creatively, explore different angles.",
                "",
                "- ULTRA-SPECIFIC function names",
                "- COLLECT PAPER TITLES for each function",
                "- TRACK INDIRECT INTERACTORS from all cascades",
                "- No constraints on search strategy!",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2b - Deep function research + paper titles + indirect tracking
    StepConfig(
        name="step2b_deep_function_research",
        model="gemini-2.5-pro",
        deep_research=True,
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
                "║  STEP 2b: DEEP FUNCTION RESEARCH + TITLES + INDIRECT         ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "Deep research for obscure/rare functions:",
                "- Target interactors with <3 functions",
                "- Recent discoveries (2020-2025)",
                "- COLLECT PAPER TITLES (no constraints!)",
                "- TRACK INDIRECT INTERACTORS from cascades",
                "",
                "Add 10-15 more functions across interactors.",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON."
        ),
    ),

    # 2b2 - GENERATE FUNCTIONS FOR INDIRECT INTERACTIONS
    StepConfig(
        name="step2b2_indirect_functions",
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
                "║  STEP 2b2: GENERATE CHAIN-CONTEXT FUNCTIONS FOR INDIRECT     ║",
                "║            INTERACTIONS                                       ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "MAIN PROTEIN: {ctx_json.main}",
                "",
                "TASK: For EACH indirect interactor WITHOUT functions, generate chain-context functions",
                "",
                "IDENTIFY INDIRECT INTERACTORS:",
                "- Look through ctx_json.interactors",
                "- Find entries where interaction_type = 'indirect'",
                "- Check if they have empty or missing functions array",
                "",
                "FOR EACH INDIRECT INTERACTOR WITHOUT FUNCTIONS:",
                "",
                "1. **Identify the full chain:**",
                "   - Query protein: {ctx_json.main}",
                "   - Parent/Upstream: upstream_interactor field",
                "   - Indirect protein: primary field",
                "   Example: p62 → LC3 → Atg5",
                "",
                "2. **Research question:**",
                "   'How does {ctx_json.main} affect {indirect_protein} through {upstream_protein}?'",
                "   'What is the biological significance of this {ctx_json.main}→{upstream}→{indirect} cascade?'",
                "",
                "3. **Generate 1-3 chain-context functions describing:**",
                "   - The CASCADING mechanism across the FULL CHAIN (not just binary interactions)",
                "   - How the query protein's effect PROPAGATES through the chain",
                "   - The BIOLOGICAL SIGNIFICANCE of this specific pathway",
                "   - Downstream effects and outcomes",
                "",
                "4. **Function structure (same as direct interactions):**",
                "   {",
                "     'function': '<ULTRA-SPECIFIC NAME of cascade function>',",
                "     'arrow': 'activates' | 'inhibits' | 'binds',",
                "     'cellular_process': '<HOW the cascade works mechanistically>',",
                "     'effect_description': '<ONE SENTENCE summary>',",
                "     'biological_consequence': [",
                "       '{ctx_json.main} affects {upstream} → {upstream} affects {indirect} → Downstream effect → Final outcome'",
                "     ],",
                "     'specific_effects': ['<Direct facts about the cascade, NO arrows>'],",
                "     'evidence': [",
                "       {'paper_title': '<Paper discussing THIS CASCADE>', 'year': <int>}",
                "     ]",
                "   }",
                "",
                "5. **Evidence collection:**",
                "   - Find papers discussing the CHAIN RELATIONSHIP (not just individual binary interactions)",
                "   - Papers should mention how the query protein affects the indirect protein through intermediates",
                "   - If direct chain papers are scarce, cite papers for individual steps + explain the cascade logic",
                "",
                "CRITICAL RULES:",
                "- ONLY process indirect interactors WITHOUT functions (don't regenerate existing ones)",
                "- Functions must describe the FULL CASCADE (Query → Parent → Indirect)",
                "- Arrow should reflect net effect of the cascade",
                "- **UNIQUE CONTENT:** Do NOT copy functions from the upstream mediator! Describe how the *indirect* protein specifically executes the downstream effect.",
                "- Ensure the biological consequence explicitly traces the path to THIS specific indirect target.",
                "- Use ULTRA-SPECIFIC function names (see FUNCTION_NAMING_RULES above)",
                "- Search freely - no constraints!",
                "",
                "EXAMPLE:",
                "If p62 → LC3 → Atg5 (Atg5 lacks functions):",
                "Function: 'Autophagosome Formation via LC3-Mediated Recruitment'",
                "Cellular Process: 'p62 binds LC3 on the phagophore membrane, which requires Atg5-Atg12-Atg16L1 complex for LC3 lipidation. The p62-LC3 interaction is only functional when Atg5 enables LC3 conjugation to PE, creating the autophagosomal membrane where p62 can dock.'",
                "Biological Consequence: ['p62 recognizes cargo → p62 requires LC3 on membrane → Atg5-Atg12 enables LC3 lipidation → LC3-II formation → p62-LC3 docking → Autophagosome maturation']",
                "",
                "OUTPUT:",
                "- Update ctx_json with functions added to indirect interactors",
                "- Return full ctx_json with updated interactors array",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON with updated ctx_json."
        ),
    ),

    # 2b3 - RESCUE DIRECT INTERACTORS WITHOUT FUNCTIONS
    StepConfig(
        name="step2b3_rescue_direct_functions",
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
                "║  STEP 2b3: RESCUE DIRECT INTERACTORS WITHOUT FUNCTIONS       ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "MAIN PROTEIN: {ctx_json.main}",
                "",
                "TASK: Find ALL direct interactors WITHOUT functions and generate functions for them",
                "",
                "CRITICAL FIRST STEP - IDENTIFY MISSED DIRECT INTERACTORS:",
                "1. SCAN ctx_json.interactors to find ALL entries matching BOTH conditions:",
                "   a) interaction_type = 'direct' (or missing/null, which implies direct)",
                "   b) functions = [] (empty array) OR functions field is missing",
                "",
                "2. CREATE AN EXPLICIT LIST of these missed interactors:",
                "   - Write out their names clearly: [Protein1, Protein2, Protein3, ...]",
                "   - Count how many you found",
                "   - This list MUST be complete - check EVERY interactor in ctx_json.interactors",
                "",
                "3. VERIFY YOUR LIST:",
                "   - Double-check you didn't skip any interactors",
                "   - Ignore interactors that already have functions (functions: [{...}])",
                "   - Ignore interactors with interaction_type = 'indirect' (handled by Step 2b2)",
                "",
                "WHY THIS STEP EXISTS:",
                "Sometimes the previous batching steps (2a-2a5) miss certain interactors due to:",
                "- LLM batching limitations (Gemini processes in chunks)",
                "- Edge cases in the differential output merge logic",
                "- Interactors added late in the process",
                "This step ensures 100% coverage - NO interactor should lack functions!",
                "",
                "FOR EACH DIRECT INTERACTOR WITHOUT FUNCTIONS:",
                "",
                "1. **Research question:**",
                "   'What are the known functional interactions between {ctx_json.main} and {interactor}?'",
                "   'How does {ctx_json.main} affect {interactor} (or vice versa)?'",
                "",
                "2. **Generate 1-3 functions describing:**",
                "   - The BINARY interaction between query and interactor",
                "   - Molecular mechanisms (binding sites, post-translational modifications, etc.)",
                "   - Cellular processes affected by this interaction",
                "   - Biological outcomes and phenotypic consequences",
                "",
                "3. **Function structure (standard format):**",
                "   {",
                "     'function': '<ULTRA-SPECIFIC NAME of cellular function>',",
                "     'arrow': 'activates' | 'inhibits' | 'binds',",
                "     'cellular_process': '<HOW the interaction works mechanistically>',",
                "     'effect_description': '<ONE SENTENCE summary>',",
                "     'biological_consequence': [",
                "       '{ctx_json.main} affects {interactor} → Intermediate step → Final outcome'",
                "     ],",
                "     'specific_effects': ['<Direct facts about the interaction, NO arrows>'],",
                "     'evidence': [",
                "       {'paper_title': '<Paper title>', 'year': <int>}",
                "     ]",
                "   }",
                "",
                "4. **Evidence collection:**",
                "   - Find papers directly discussing {ctx_json.main} ↔ {interactor} interaction",
                "   - Papers should describe functional effects (not just co-localization)",
                "   - Prioritize papers with mechanistic detail",
                "",
                "CRITICAL RULES:",
                "- ONLY process direct interactors WITHOUT functions (don't regenerate existing ones)",
                "- Functions must describe the DIRECT interaction (not cascades)",
                "- Use ULTRA-SPECIFIC function names (see FUNCTION_NAMING_RULES above)",
                "- If an interactor has interaction_type='indirect', SKIP IT (Step 2b2 handles those)",
                "- Search freely - no constraints!",
                "",
                "EXAMPLE:",
                "If ATXN3 ↔ G3BP1 (G3BP1 lacks functions):",
                "Function: 'Stress Granule Formation and Mutant ATXN3 Sequestration'",
                "Cellular Process: 'G3BP1, as a stress granule nucleator, reduces the levels and aggregation of mutant ATXN3 by sequestering it into stress granules during cellular stress, preventing toxic accumulation in the nucleoplasm.'",
                "Biological Consequence: ['Cellular stress → G3BP1 nucleates stress granules → Mutant ATXN3 recruited to stress granules → Reduced toxic aggregation → Neuroprotection']",
                "",
                "MANDATORY VALIDATION (DO NOT SKIP!):",
                "Before returning your output, you MUST verify:",
                "",
                "1. COUNT: How many direct interactors had empty functions[] at the start?",
                "   - List them by name: [Protein1, Protein2, ...]",
                "",
                "2. VERIFY: Did you generate functions for EVERY single one?",
                "   - Check each protein from your list",
                "   - Confirm functions array is no longer empty for each",
                "",
                "3. FINAL CHECK: Scan ctx_json.interactors one more time:",
                "   - NO direct interactor should have functions: []",
                "   - If you find any, you MUST add functions for them NOW",
                "",
                "⚠️  CRITICAL: This is your LAST CHANCE to add functions before arrow determination!",
                "   Any interactor without functions will default to arrow='binds' (which is often wrong!)",
                "",
                "OUTPUT:",
                "- Update ctx_json with functions added to ALL direct interactors without functions",
                "- Return full ctx_json with updated interactors array",
                "- Ensure 100% completion - every direct interactor MUST have at least 1 function",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON with updated ctx_json."
        ),
    ),

    # 2c - ARROW DETERMINATION (PER-INTERACTOR) - TEMPLATE
    # NOTE: config_dynamic.py will generate ONE of these for EACH interactor
    # This is a TEMPLATE showing the logic - actual steps generated dynamically
    StepConfig(
        name="step2c_arrow_TEMPLATE",
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
            + "\n".join([
                "╔═══════════════════════════════════════════════════════════════╗",
                "║  STEP 2c: ARROW DETERMINATION FOR {INTERACTOR}               ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "NOW determine arrows/directions for: {INTERACTOR}",
                "",
                "QUERY PROTEIN: {ctx_json.main}",
                "INTERACTOR: {INTERACTOR}",
                "",
                "FUNCTIONS AFFECTED BY THEIR INTERACTION:",
                "{FUNCTIONS_LIST}",
                "",
                "⚠️  CRITICAL: You must determine TWO SEPARATE effects for EACH function:",
                "1. INTERACTION EFFECT: Effect on the DOWNSTREAM PROTEIN in the interaction",
                "2. FUNCTION EFFECT: Effect on the FUNCTION itself (already provided, verify accuracy)",
                "",
                "YOUR TASK FOR EACH FUNCTION:",
                "",
                "STEP 1: Determine INTERACTION DIRECTION",
                "- Which protein affects which in THIS specific function context?",
                "- Options: 'main_to_primary' (Query→Interactor) or 'primary_to_main' (Interactor→Query)",
                "",
                "STEP 2: Determine INTERACTION EFFECT (on the downstream PROTEIN)",
                "- Research the NORMAL ROLE of the downstream protein",
                "- Ask: Does this interaction ENHANCE or OPPOSE that protein's normal function?",
                "- Set interaction_effect: 'activates', 'inhibits', or 'regulates'",
                "",
                "STEP 3: Verify FUNCTION EFFECT (on the FUNCTION itself)",
                "- The function arrow is already set from Step 2a",
                "- Verify it's correct based on your research",
                "- This can DIFFER from interaction_effect!",
                "",
                "═══════════════════════════════════════════════════════════════",
                "EXAMPLE 1: ATXN3-VCP interaction affects ERAD",
                "═══════════════════════════════════════════════════════════════",
                "",
                "Current function entry: {'function': 'ERAD', 'arrow': 'inhibits'}",
                "",
                "ANALYSIS:",
                "1. INTERACTION DIRECTION:",
                "   → ERAD is VCP's normal function (not ATXN3's)",
                "   → So this describes ATXN3 → VCP interaction",
                "   → Direction: 'main_to_primary'",
                "",
                "2. INTERACTION EFFECT (on VCP protein):",
                "   → Research: VCP normally ACTIVATES ERAD",
                "   → Result: ERAD is INHIBITED (function arrow = 'inhibits')",
                "   → Logic: If VCP normally activates ERAD, but ERAD is inhibited,",
                "            then ATXN3 must be INHIBITING VCP's normal activity",
                "   → interaction_effect: 'inhibits'",
                "",
                "3. FUNCTION EFFECT (on ERAD itself):",
                "   → Function arrow already says 'inhibits'",
                "   → This is correct: ERAD is inhibited",
                "   → Keep arrow: 'inhibits'",
                "",
                "OUTPUT FOR THIS FUNCTION:",
                "{",
                "  'function': 'ERAD',",
                "  'interaction_direction': 'main_to_primary',  // ATXN3 → VCP",
                "  'interaction_effect': 'inhibits',  // ATXN3 inhibits VCP",
                "  'arrow': 'inhibits',  // ERAD is inhibited",
                "  ...",
                "}",
                "",
                "VISUAL: [Inhibits] ATXN3 → VCP  ||  ERAD [Inhibited]",
                "        ^^^^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^",
                "        interaction_effect           function arrow",
                "",
                "═══════════════════════════════════════════════════════════════",
                "EXAMPLE 2: VCP-ATXN3 interaction affects Autophagy",
                "═══════════════════════════════════════════════════════════════",
                "",
                "Current function entry: {'function': 'Autophagy', 'arrow': 'inhibits'}",
                "",
                "ANALYSIS:",
                "1. INTERACTION DIRECTION:",
                "   → Autophagy is ATXN3's function (being affected)",
                "   → So this describes VCP → ATXN3 interaction",
                "   → Direction: 'primary_to_main'",
                "",
                "2. INTERACTION EFFECT (on ATXN3 protein):",
                "   → Research: ATXN3 normally ACTIVATES Autophagy",
                "   → Result: Autophagy is INHIBITED (function arrow = 'inhibits')",
                "   → Logic: If ATXN3 normally activates Autophagy, but Autophagy is inhibited,",
                "            then VCP must be INHIBITING ATXN3's normal activity",
                "   → interaction_effect: 'inhibits'",
                "",
                "3. FUNCTION EFFECT (on Autophagy itself):",
                "   → Function arrow already says 'inhibits'",
                "   → This is correct: Autophagy is inhibited",
                "   → Keep arrow: 'inhibits'",
                "",
                "OUTPUT FOR THIS FUNCTION:",
                "{",
                "  'function': 'Autophagy',",
                "  'interaction_direction': 'primary_to_main',  // VCP → ATXN3",
                "  'interaction_effect': 'inhibits',  // VCP inhibits ATXN3",
                "  'arrow': 'inhibits',  // Autophagy is inhibited",
                "  ...",
                "}",
                "",
                "VISUAL: [Inhibits] VCP → ATXN3  ||  Autophagy [Inhibited]",
                "        ^^^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^^",
                "        interaction_effect          function arrow",
                "",
                "═══════════════════════════════════════════════════════════════",
                "EXAMPLE 3: When effects DIFFER (inhibit protein, activate function)",
                "═══════════════════════════════════════════════════════════════",
                "",
                "Scenario: Protein A inhibits Protein B, which normally REPRESSES Function X",
                "Result: Function X is now RELEASED from repression → activated!",
                "",
                "OUTPUT:",
                "{",
                "  'function': 'Function X',",
                "  'interaction_direction': 'main_to_primary',",
                "  'interaction_effect': 'inhibits',  // A inhibits B (protein level)",
                "  'arrow': 'activates',  // Function X is activated (function level)",
                "  ...",
                "}",
                "",
                "VISUAL: [Inhibits] A → B  ||  Function X [Activated]",
                "         ^^^^^^^^                        ^^^^^^^^^^",
                "         Different effects! B is inhibited, Function is activated",
                "",
                "═══════════════════════════════════════════════════════════════",
                "KEY RULES:",
                "═══════════════════════════════════════════════════════════════",
                "",
                "1. interaction_effect describes effect on the DOWNSTREAM PROTEIN",
                "2. arrow (function effect) describes effect on the FUNCTION",
                "3. These CAN BE DIFFERENT! (inhibit protein → activate function)",
                "4. For bidirectional functions, create TWO function entries (one per direction)",
                "5. Always research the downstream protein's NORMAL ROLE before deciding",
                "",
                "OUTPUT STRUCTURE:",
                "Update EACH function in the interactor's functions array with:",
                "{",
                "  'function': '<function name>',",
                "  'interaction_direction': 'main_to_primary' | 'primary_to_main',",
                "  'interaction_effect': 'activates' | 'inhibits' | 'regulates',",
                "  'arrow': 'activates' | 'inhibits',  // Keep existing or correct if wrong",
                "  'cellular_process': '...',",
                "  'effect_description': '...',",
                "  ...",
                "}",
                "",
                "Also add interactor-level summary:",
                "{",
                "  'primary': '{INTERACTOR}',",
                "  'direction': 'bidirectional' | 'main_to_primary' | 'primary_to_main',",
                "  'arrow': 'activates' | 'inhibits' | 'regulates',  // Aggregate if needed",
                "  'intent': '<mechanism>',  // e.g., 'phosphorylation', 'ubiquitination'",
                "  'functions': [<updated function entries with interaction_effect>]",
                "}",
                "",
                "Search freely to understand both proteins' normal biological roles!",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON with updated arrow/direction/intent for {INTERACTOR}"
        ),
    ),

    # 2g - Final quality control
    StepConfig(
        name="step2g_final_qc",
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
            + "\n".join([
                "╔═══════════════════════════════════════════════════════════════╗",
                "║  STEP 2g: FINAL QUALITY CONTROL                              ║",
                "╚═══════════════════════════════════════════════════════════════╝",
                "",
                "FINAL CHECKS:",
                "",
                "1. **Arrows/directions determined for ALL interactors** (set in Step 2c)",
                "2. **Each function has complete field descriptions**",
                "3. **Biological cascades are detailed and multiple where appropriate**",
                "4. **All function fields are well-described**",
                "5. **Function names are specific and clear**",
                "6. **Indirect interactors tracked with upstream_interactor set**",
                "7. **Paper titles collected for functions**",
                "",
                "Spot-check 5-10 random entries for accuracy.",
                "Note: Evidence validation happens in claim_fact_checker.py next.",
            ])
            + "\n\n"
            + SCHEMA_HELP
            + "\n\n"
            + "Return ONLY JSON with step_json={'step':'step2g_final_qc','status':'validated'}"
        ),
    ),

    # 3 - Snapshot (handled by runner)
    StepConfig(
        name="step3_snapshot",
        model="gemini-2.5-pro",
        deep_research=False,
        reasoning_effort="high",
        use_google_search=False,
        thinking_budget=None,  # Not used for snapshot
        max_output_tokens=MAX_OUTPUT_TOKENS,
        expected_columns=["ctx_json", "snapshot_json", "ndjson", "step_json"],
        system_prompt=None,
        prompt_template="Snapshot handled by runner.",
    ),
]
