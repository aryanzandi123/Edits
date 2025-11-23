# pruner.py
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .llm_response_parser import extract_json_from_llm_response

# ---------- Public config ----------
HARD_MAX_KEEP_DEFAULT = 20
PRUNED_DIRNAME = "pruned"  # under CACHE_DIR

# ---------- Util / validation ----------

PROTEIN_RE = re.compile(r"^[A-Za-z0-9._-]+$")

def norm_symbol(s: str) -> str:
    return (s or "").strip().upper()

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _years_from_evidence(interactor: dict) -> List[int]:
    years: Set[int] = set()

    for ev in interactor.get("evidence", []) or []:
        y = ev.get("year")
        if isinstance(y, int):
            years.add(y)

    for fn in interactor.get("functions", []) or []:
        for ev in fn.get("evidence", []) or []:
            y = ev.get("year")
            if isinstance(y, int):
                years.add(y)

    ys = sorted(years)
    return ys

def _pmids_count(interactor: dict) -> int:
    n = 0
    ip = interactor.get("pmids")
    if isinstance(ip, list):
        n += len(ip)
    for fn in interactor.get("functions", []) or []:
        if isinstance(fn.get("pmids"), list):
            n += len(fn["pmids"])
    return n

def _functions_preview(interactor: dict, k: int = 3) -> List[str]:
    names: List[str] = []
    for fn in interactor.get("functions", []) or []:
        nm = fn.get("function")
        if isinstance(nm, str) and nm.strip():
            names.append(nm.strip())
        if len(names) >= k:
            break
    return names

def _tokenize_for_overlap(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    # lowercase alphanumerics
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    # simple stopword trimming
    stops = {"the","and","or","if","of","to","in","on","for","by","with","a","an","is","are","was","were"}
    return {t for t in toks if t not in stops and len(t) >= 3}

# ---------- Candidate pack ----------

def build_candidate_pack(
    full_json: dict,
    current_nodes: Optional[List[str]],
    parent_edge: Optional[dict],
) -> Tuple[str, List[dict]]:
    """
    Returns (main_symbol, candidates[list]) where each candidate is a skinny object
    for prompting the LLM (or fallback scoring).
    Handles both old format (interactors) and new format (proteins + interactions).
    """
    snap = full_json.get("snapshot_json") or full_json
    main = norm_symbol(snap.get("main") or full_json.get("main") or full_json.get("primary") or "")
    current_set = {norm_symbol(s) for s in (current_nodes or [])}

    # Extract overlap hints from the parent_edge fields (intent/summary/arrow)
    pe_intent = (parent_edge or {}).get("intent") or ""
    pe_arrow  = (parent_edge or {}).get("arrow") or ""
    pe_sum    = (parent_edge or {}).get("support_summary") or ""
    pe_tokens = _tokenize_for_overlap(" ".join([str(pe_intent), str(pe_arrow), str(pe_sum)]))

    candidates: List[dict] = []

    # NEW FORMAT: proteins + interactions
    if "proteins" in snap and "interactions" in snap:
        proteins = snap.get("proteins", [])
        interactions = snap.get("interactions", [])

        # For each protein (except main), find its interaction data
        for protein in proteins:
            sym = norm_symbol(protein)
            if not sym or sym == norm_symbol(main):
                continue  # Skip main protein

            # Find interaction involving this protein and main
            interaction_data = None
            for interaction in interactions:
                source = norm_symbol(interaction.get("source", ""))
                target = norm_symbol(interaction.get("target", ""))
                # Check if this interaction connects main and this protein
                if (source == sym and target == norm_symbol(main)) or \
                   (source == norm_symbol(main) and target == sym):
                    interaction_data = interaction
                    break

            # Build candidate from interaction data (or defaults if not found)
            it = interaction_data or {}
            cand = {
                "primary": sym,
                "confidence": it.get("confidence", 0.0),
                "pmids_count": _pmids_count(it),
                "arrow": (it.get("arrow") or "").lower(),
                "intent": (it.get("intent") or "").lower(),
                "direction": (it.get("direction") or "").lower(),
                "functions_preview": _functions_preview(it, k=3),
                "years": _years_from_evidence(it),
                "overlaps_current_graph": (sym in current_set),
                "mechanism_overlap_hint": False,
                # Chain metadata for indirect interactors
                "interaction_type": it.get("interaction_type", "direct"),
                "upstream_interactor": it.get("upstream_interactor"),
                "mediator_chain": it.get("mediator_chain", []),
                "depth": it.get("depth", 1),
            }

            # Mechanism overlap hint
            if pe_tokens and cand["functions_preview"]:
                fn_text = " ".join(cand["functions_preview"])
                fn_tokens = _tokenize_for_overlap(fn_text)
                if fn_tokens & pe_tokens:
                    cand["mechanism_overlap_hint"] = True

            candidates.append(cand)

    # OLD FORMAT: interactors (legacy support)
    else:
        interactors = snap.get("interactors") or []
        for it in interactors:
            sym = norm_symbol(it.get("primary") or it.get("label") or it.get("symbol") or "")
            if not sym:
                continue
            cand = {
                "primary": sym,
                "confidence": it.get("confidence", 0.0),
                "pmids_count": _pmids_count(it),
                "arrow": (it.get("arrow") or "").lower(),
                "intent": (it.get("intent") or "").lower(),
                "direction": (it.get("direction") or "").lower(),
                "functions_preview": _functions_preview(it, k=3),
                "years": _years_from_evidence(it),
                "overlaps_current_graph": (sym in current_set),
                "mechanism_overlap_hint": False,
                # Chain metadata for indirect interactors
                "interaction_type": it.get("interaction_type", "direct"),
                "upstream_interactor": it.get("upstream_interactor"),
                "mediator_chain": it.get("mediator_chain", []),
                "depth": it.get("depth", 1),
            }

            # Mechanism overlap hint
            if pe_tokens and cand["functions_preview"]:
                fn_text = " ".join(cand["functions_preview"])
                fn_tokens = _tokenize_for_overlap(fn_text)
                if fn_tokens & pe_tokens:
                    cand["mechanism_overlap_hint"] = True

            candidates.append(cand)

    return main, candidates

# ---------- LLM wiring (optionally used) ----------

def build_pruning_prompt(
    parent: str,
    protein: str,
    main: str,
    candidates: List[dict],
    hard_max_keep: int,
    current_nodes: List[str],
) -> str:
        """
        Master prompt for relevance-first pruning (Option B).
        Returns a single string instruction. Output must be strict JSON:
        {
            "keep": ["SYMBOL1","SYMBOL2", ...],   // HGNC uppercase; subset of candidates; length <= hard_max_keep
            "reasons": { "SYMBOL1": "≤120 char reason", ... }
        }
        """
        P0 = norm_symbol(parent)    # parent/query protein (graph root)
        P1 = norm_symbol(protein)   # clicked interactor being expanded
        MAIN = norm_symbol(main)    # main inside expanded JSON (should equal P1)
        cur_nodes = [norm_symbol(x) for x in (current_nodes or [])]

        # Embed a compact, self-contained candidate pack for the model
        pack = {
            "parent": P0,
            "protein": P1,
            "main_in_file": MAIN,
            "hard_max_keep": int(hard_max_keep),
            "current_nodes": cur_nodes,
            "candidates": candidates,   # already skinny; contains only selection cues
        }
        pack_json = json.dumps(pack, ensure_ascii=False, indent=2)

        prompt = f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║   PRUNING STAGE – RELEVANCE-FIRST SELECTION (NO SCHEMA MODIFICATION) ║
    ╚══════════════════════════════════════════════════════════════════════╝
    ROLE: You are an expert molecular biologist curating the MOST RELEVANT subset
            of interactors to keep when expanding a subgraph in a protein network.

    CRITICAL CONTEXT:
        • Parent/Query protein (graph root): {P0}
        • Interactor being expanded (clicked): {P1}
        • Hard maximum KEEP count: {hard_max_keep}
        • Current visible nodes (to consider for network relevance): {cur_nodes}

    OBJECTIVE (RELEVANCE FIRST, QUALITY OVER SPEED):
        Select ONLY those candidate interactors whose biology is MOST RELEVANT to:
        (1) the specific {P0} → {P1} relationship (its mechanism & effect)
        (2) the current visible graph context (prefer bridging/triangles)
        — Bias to newer/nicher when it aligns with mechanism, but do NOT sacrifice
        mechanistic relevance or evidence quality.
        — You are NOT rewriting evidence or schema. You are only CHOOSING which
        interactors to keep from the provided CANDIDATE PACK.

    DO NOT BROWSE OR SEARCH. Use ONLY the structured cues provided below.

    ────────────────────────────────────────────────────────────────────────
    STRICT NON-HALLUCINATION & OUTPUT RULES (ABSOLUTE)
    ────────────────────────────────────────────────────────────────────────
        1) Output STRICT JSON ONLY (no prose, no markdown).
        2) "keep" must be an array of HGNC symbols (UPPERCASE), each PRESENT in
            candidates[].primary. Do NOT invent or rename symbols.
        3) |keep| ≤ {hard_max_keep}. If fewer than {hard_max_keep} are truly relevant,
            return fewer (quality first).
        4) "reasons" must map EACH kept symbol → a ≤120 char justification focused on
            mechanism/context relevance (not generic praise).
        5) Do NOT modify candidate fields; do NOT add evidence; do NOT re-score.
        6) If a candidate lacks data, treat it as weaker unless mechanistic fit is
            clearly superior from provided function names/intents.

    ────────────────────────────────────────────────────────────────────────
    SELECTION POLICY (RELEVANCE HEURISTICS YOU MUST APPLY)
    ────────────────────────────────────────────────────────────────────────
    HIGH-PRIORITY SIGNALS (most important):
        • MECHANISM FIT: functions_preview overlap with the {P0}→{P1} mechanism.
        (The pack provides "mechanism_overlap_hint". Use it when true.)
        • CONTEXT FIT: overlaps_current_graph = true (bridges to current nodes),
        or likely to close triangles with {P0}, {P1}, or other visible nodes.
        • DIRECTIONALITY: when available, prefer mechanistically interpretable
        'activates'/'inhibits' edges over neutral 'binds' if relevance is tied.

    SECONDARY SIGNALS (tie-breakers):
        • pmids_count (more independent evidence is better).
        • confidence (higher is better).
        • recency cue from years (prefer newer when mechanism/context tie).

    NEGATIVE SIGNALS (de-prioritize unless mechanism fit is compelling):
        • Pure binding with no functional context (arrow='binds') AND no overlap.
        • Redundant function themes already strongly covered in the visible graph.

    ────────────────────────────────────────────────────────────────────────
    DIVERSITY & COVERAGE (WHEN POSSIBLE)
    ────────────────────────────────────────────────────────────────────────
        • Prefer a set that spans complementary function themes (e.g., ERAD,
        proteostasis, autophagy, DNA repair) WHEN these themes are relevant to
        {P0}→{P1}. Avoid 10 near-duplicates of the same function label.
        • However, never include a weakly relevant item just for diversity.

    ────────────────────────────────────────────────────────────────────────
    FAIL-SAFE BEHAVIOR
    ────────────────────────────────────────────────────────────────────────
        • If very few items are clearly relevant, return FEWER than {hard_max_keep}.
        • NEVER exceed {hard_max_keep}. NEVER include symbols outside candidates.

    ────────────────────────────────────────────────────────────────────────
    OUTPUT FORMAT (STRICT JSON ONLY)
    ────────────────────────────────────────────────────────────────────────
    {{
        "keep": ["HGNC1","HGNC2", "..."],          // length ≤ {hard_max_keep}; subset of candidates[].primary
        "reasons": {{
            "HGNC1": "≤120 char mechanism/context reason",
            "HGNC2": "≤120 char mechanism/context reason"
        }}
    }}

    EXAMPLES OF GOOD REASONS (STYLE, not content):
        • "Bridges ERAD with VCP; functions match proteostasis; strong evidence"
        • "Autophagy link downstream of {P1}; closes triangle with current nodes"
        • "Directional inhibition aligns with {P0}→{P1} mechanism; recent support"

    DO NOT RETURN MORE THAN {hard_max_keep}. DO NOT RETURN MARKDOWN.

    ────────────────────────────────────────────────────────────────────────
    CANDIDATE PACK (READ CAREFULLY; THIS IS YOUR ONLY EVIDENCE)
    ────────────────────────────────────────────────────────────────────────
    {pack_json}
    """
        return prompt

def _call_gemini_json(prompt: str, api_key: str, max_retries: int = 3) -> dict:
    """
    Call Gemini 2.5 Pro without Google Search (fast), parse strict JSON.
    """
    from google import genai as google_genai
    from google.genai import types

    client = google_genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        max_output_tokens=2048,
        temperature=0.0,
        top_p=0.4,
        tools=[],  # no search for speed
    )

    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config,
            )
            if hasattr(resp, "text") and resp.text:
                return extract_json_from_llm_response(resp.text)
            if hasattr(resp, "candidates") and resp.candidates:
                parts = resp.candidates[0].content.parts
                out = "".join(p.text for p in parts if hasattr(p, "text"))
                return extract_json_from_llm_response(out)
            raise RuntimeError("Empty model response")
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"LLM call failed: {last_err}")

def llm_select_keep_list(
    api_key: Optional[str],
    parent: str,
    protein: str,
    main: str,
    candidates: List[dict],
    hard_max_keep: int,
    current_nodes: List[str],
    use_llm: bool,
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """
    Returns (keep_list, reasons_map, llm_error)
    If use_llm=False or any failure occurs, returns fallback selection with llm_error set.
    """
    if use_llm:
        try:
            prompt = build_pruning_prompt(parent, protein, main, candidates, hard_max_keep, current_nodes)
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not configured.")
            result = _call_gemini_json(prompt, api_key)
            keep = result.get("keep") or []
            reasons = result.get("reasons") or {}
            # Normalize and validate
            keep_norm = []
            cand_set = {c["primary"] for c in candidates}
            for sym in keep:
                s = norm_symbol(sym)
                if s and s in cand_set and s not in keep_norm:
                    keep_norm.append(s)
                if len(keep_norm) >= hard_max_keep:
                    break
            if keep_norm:
                return keep_norm, {norm_symbol(k): v for k, v in reasons.items()}, None
            # fallthrough to fallback if empty
            err = "LLM returned empty/invalid keep list; using fallback."
        except NotImplementedError:
            err = "Pruning prompt not implemented; using fallback."
        except Exception as e:
            err = f"LLM error: {e}; using fallback."
    else:
        err = "LLM disabled; using fallback."

    # Fallback: relevance-first heuristic (mechanism overlap, graph overlap), then evidence/confidence
    scored: List[Tuple[float, str]] = []
    for c in candidates:
        score = 0.0
        if c.get("mechanism_overlap_hint"):
            score += 3.0
        if c.get("overlaps_current_graph"):
            score += 2.0
        score += min(2.0, (c.get("pmids_count") or 0) * 0.15)
        score += min(1.0, (c.get("confidence") or 0.0))
        # Very mild recency nudge (but relevance dominates)
        ys = c.get("years") or []
        if ys:
            latest = max(ys)
            # +0.2 if >= 2021, else 0
            if latest >= 2021:
                score += 0.2
        scored.append((score, c["primary"]))
    scored.sort(key=lambda t: t[0], reverse=True)

    keep = [s for _, s in scored[:hard_max_keep]]
    reasons = {s: "Selected by fallback relevance heuristic (mechanism/graph overlap, evidence, confidence)." for s in keep}
    return keep, reasons, err

# ---------- Prune builder ----------

def build_pruned_json(
    full_json: dict,
    keep: List[str],
    parent: str,
    protein: str,
    reasons: Dict[str, str],
    hard_max_keep: int,
) -> dict:
    """
    Returns a pruned JSON with only the kept interactors in snapshot_json.
    Handles both old format (interactors) and new format (proteins + interactions).
    ctx_json is preserved as-is (safer & simpler).
    """
    snap = full_json.get("snapshot_json") or full_json
    keep_set = {norm_symbol(s) for s in keep}

    # NEW FORMAT: proteins + interactions
    if "proteins" in snap and "interactions" in snap:
        # Filter proteins: keep main protein + kept interactors
        main_protein = snap.get("main", protein)
        pruned_proteins = [main_protein]  # Always include main
        for p in snap.get("proteins", []):
            if norm_symbol(p) in keep_set:
                pruned_proteins.append(p)

        # Filter interactions: only those involving kept proteins
        pruned_protein_set = {norm_symbol(p) for p in pruned_proteins}
        pruned_interactions = []
        for interaction in snap.get("interactions", []):
            source = norm_symbol(interaction.get("source", ""))
            target = norm_symbol(interaction.get("target", ""))
            if source in pruned_protein_set and target in pruned_protein_set:
                pruned_interactions.append(interaction)

        out = {
            "snapshot_json": {
                "main": main_protein,
                "proteins": pruned_proteins,
                "interactions": pruned_interactions,
            },
            "ctx_json": full_json.get("ctx_json"),
            "_prune_meta": {
                "parent": parent,
                "protein": protein,
                "keep_count": len(keep),
                "hard_max_keep": hard_max_keep,
                "reasons": reasons,
                "model": "gemini-2.5-pro",
                "created_at": int(time.time()),
            },
        }
    # OLD FORMAT: interactors (legacy support)
    else:
        interactors = snap.get("interactors") or []
        pruned_interactors = []
        for it in interactors:
            sym = norm_symbol(it.get("primary") or it.get("label") or it.get("symbol") or "")
            if sym in keep_set:
                pruned_interactors.append(it)

        out = {
            "snapshot_json": {
                **{k: v for k, v in snap.items() if k != "interactors"},
                "interactors": pruned_interactors,
            },
            "ctx_json": full_json.get("ctx_json"),
            "_prune_meta": {
                "parent": parent,
                "protein": protein,
                "keep_count": len(keep),
                "hard_max_keep": hard_max_keep,
                "reasons": reasons,
                "model": "gemini-2.5-pro",
                "created_at": int(time.time()),
            },
        }

    return out


def preserve_indirect_chains(keep: List[str], full_data: dict, reasons: dict) -> List[str]:
    """
    Post-process keep list to preserve indirect interaction chain integrity.

    Rules:
    1. If an indirect interactor is kept, ensure its upstream is also kept
    2. If an upstream is pruned, remove its indirect children from keep list

    Args:
        keep: List of protein symbols to keep
        full_data: Full JSON data with all interactors
        reasons: Reasons dict for kept interactors (updated in-place)

    Returns:
        Updated keep list with chain integrity preserved
    """
    snapshot = full_data.get("snapshot_json", full_data)
    interactors = snapshot.get("interactors", [])

    # Build map of protein -> interactor data
    interactor_map = {norm_symbol(i.get("primary", "")): i for i in interactors if i.get("primary")}

    # Find indirect interactions and their upstreams
    indirect_deps = {}  # {indirect_protein: upstream_protein}
    for protein, data in interactor_map.items():
        if data.get("interaction_type") == "indirect" and data.get("upstream_interactor"):
            upstream = norm_symbol(data.get("upstream_interactor"))
            indirect_deps[protein] = upstream

    # Pass 1: Add missing upstreams for kept indirect interactors
    added = []
    for protein in list(keep):
        if protein in indirect_deps:
            upstream = indirect_deps[protein]
            if upstream not in keep and upstream in interactor_map:
                keep.append(upstream)
                added.append(upstream)
                reasons[upstream] = f"Required upstream for indirect interactor {protein}"

    # Pass 2: Remove orphaned indirect interactors (whose upstream was pruned)
    removed = []
    for protein in list(keep):
        if protein in indirect_deps:
            upstream = indirect_deps[protein]
            if upstream not in keep:
                keep.remove(protein)
                removed.append(protein)
                if protein in reasons:
                    del reasons[protein]

    if added or removed:
        print(f"Chain-aware pruning: added {len(added)} upstreams, removed {len(removed)} orphaned indirect", file=sys.stderr)

    return keep


# ---------- Main entry invoked by app thread ----------

def run_prune_job(
    full_json_path: Path,
    pruned_json_path: Path,
    parent: str,
    current_nodes: Optional[List[str]] = None,
    parent_edge: Optional[dict] = None,
    hard_max_keep: int = HARD_MAX_KEEP_DEFAULT,
    api_key: Optional[str] = None,
    use_llm: bool = False,
) -> dict:
    """
    End-to-end pruning execution.
    - Loads full cached JSON for the expanded protein.
    - Builds candidates.
    - Uses LLM (if enabled) or fallback to select keep-list (max hard_max_keep).
    - Writes pruned JSON to pruned_json_path.
    Returns the pruned dict.
    """
    parent = norm_symbol(parent)
    if not parent or not PROTEIN_RE.match(parent):
        raise ValueError("Invalid parent symbol.")

    # Read snapshot from main file
    snapshot_data = json.loads(Path(full_json_path).read_text(encoding="utf-8"))

    # Read ctx_json from metadata file (if it exists)
    metadata_path = full_json_path.parent / f"{full_json_path.stem}_metadata.json"
    ctx_data = {}
    if metadata_path.exists():
        try:
            ctx_file = json.loads(metadata_path.read_text(encoding="utf-8"))
            ctx_data = ctx_file.get("ctx_json", {})
        except Exception:
            pass  # If metadata doesn't exist or is invalid, continue without it

    # Combine both for processing
    full_data = {
        "snapshot_json": snapshot_data.get("snapshot_json", {}),
        "ctx_json": ctx_data
    }

    protein = norm_symbol((full_data.get("snapshot_json") or full_data).get("main") or full_data.get("main") or "")
    if not protein:
        raise ValueError("Expanded JSON missing 'main' protein symbol.")

    main, candidates = build_candidate_pack(full_data, current_nodes or [], parent_edge or {})

    keep, reasons, llm_error = llm_select_keep_list(
        api_key=api_key,
        parent=parent,
        protein=protein,
        main=main,
        candidates=candidates,
        hard_max_keep=hard_max_keep,
        current_nodes=[norm_symbol(s) for s in (current_nodes or [])],
        use_llm=use_llm,
    )

    # Post-process keep list to preserve indirect interaction chains
    keep = preserve_indirect_chains(keep, full_data, reasons)

    pruned = build_pruned_json(
        full_json=full_data,
        keep=keep,
        parent=parent,
        protein=protein,
        reasons=reasons,
        hard_max_keep=hard_max_keep,
    )

    # Record error if any
    if llm_error:
        pruned["_prune_meta"]["llm_error"] = llm_error

    pruned_json_path.parent.mkdir(parents=True, exist_ok=True)
    Path(pruned_json_path).write_text(json.dumps(pruned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pruned

# ---------- Helpers used by app ----------

def pruned_filename(parent: str, protein: str) -> str:
    return f"{norm_symbol(parent)}_for_{norm_symbol(protein)}.json"

def is_pruned_fresh(full_path: Path, pruned_path: Path, hard_max_keep: int) -> bool:
    if not pruned_path.exists():
        return False
    try:
        pj = json.loads(pruned_path.read_text(encoding="utf-8"))
        meta = pj.get("_prune_meta") or {}
        if int(meta.get("hard_max_keep") or 0) != int(hard_max_keep):
            return False
    except Exception:
        return False
    return pruned_path.stat().st_mtime >= full_path.stat().st_mtime

def parse_prune_job_id(job_id: str) -> Tuple[str, str]:
    """
    job_id format: prune:PARENT:PROTEIN
    Returns (parent, protein)
    """
    parts = (job_id or "").split(":")
    if len(parts) != 3 or parts[0] != "prune":
        raise ValueError("Invalid prune job id.")
    return norm_symbol(parts[1]), norm_symbol(parts[2])

def make_prune_job_id(parent: str, protein: str) -> str:
    return f"prune:{norm_symbol(parent)}:{norm_symbol(protein)}"
