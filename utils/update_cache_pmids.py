#!/usr/bin/env python3
"""Update cached JSON files with PubMed IDs inferred from evidence titles."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from utils.pubmed_match import (  # type: ignore
    DEFAULT_API_KEY,
    DEFAULT_EMAIL,
    DEFAULT_RETMAX,
    DEFAULT_SLEEP,
    Match,
    PubMedClient,
    best_match,
    )


def unique_sequence(items: Iterable[Optional[str]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def update_function_pmids(
    fn_block: Dict[str, object], client: PubMedClient, retmax: int, dry_run: bool
) -> Tuple[bool, bool]:
    evidence = fn_block.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False, True

    updated = False
    matched_pmids: List[str] = []
    matched_evidence: List[Dict[str, object]] = []
    removed_titles: List[str] = []

    for idx, record in enumerate(evidence):
        if not isinstance(record, dict):
            continue
        title = record.get("paper_title")
        if not isinstance(title, str) or not title.strip():
            continue

        try:
            ids = client.search_ids(title, retmax)
            titles = client.fetch_titles(ids)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[warn] Failed to refresh PMID for title index {idx}: {exc}", file=sys.stderr)
            continue

        match: Match = best_match(title, titles)
        pmid = match.pmid

        # Only delete if NO papers were found at all (paper doesn't exist)
        # Keep best-match PMID even if inexact
        if pmid:
            # Found a match (exact or best-match)
            if record.get("pmid") != pmid:
                print(f"[info] Updating evidence PMID '{record.get('pmid')}' -> '{pmid}' for title: {title}")
                if not dry_run:
                    record["pmid"] = pmid
                updated = True
            matched_pmids.append(pmid)
            matched_evidence.append(record)
        elif not titles:
            # No papers found at all - paper doesn't exist, should be removed
            removed_titles.append(title)
            print(f"[warn] No papers found for title '{title}' - will be removed")
        else:
            # Papers exist but no good match - keep original if it has a PMID
            existing_pmid = record.get("pmid")
            if existing_pmid:
                print(f"[info] No exact match for '{title}', keeping existing PMID: {existing_pmid}")
                matched_pmids.append(str(existing_pmid))
                matched_evidence.append(record)
            else:
                # No existing PMID and no match found - remove
                removed_titles.append(title)
                print(f"[warn] No PMID match for '{title}' and no existing PMID - will be removed")

    if removed_titles:
        print(f"[info] Removing evidence without PMID for function '{fn_block.get('function', 'UNKNOWN')}': {removed_titles}")
        updated = True

    if not matched_pmids:
        print(f"[info] Removing function '{fn_block.get('function', 'UNKNOWN')}' due to missing PMIDs.")
        return True, False

    new_pmids = unique_sequence(matched_pmids)
    if not dry_run:
        fn_block["evidence"] = matched_evidence
        if fn_block.get("pmids") != new_pmids:
            print(f"[info] Updating pmids list {fn_block.get('pmids')} -> {new_pmids}")
            fn_block["pmids"] = new_pmids
            updated = True
    else:
        if fn_block.get("pmids") != new_pmids or removed_titles:
            updated = True

    return updated, True


def collect_interactor_lists(node: object, results: List[List[Dict[str, object]]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "interactors" and isinstance(value, list):
                results.append(value)
            else:
                collect_interactor_lists(value, results)
    elif isinstance(node, list):
        for item in node:
            collect_interactor_lists(item, results)


def update_payload_pmids(
    payload: Dict[str, object],
    email: Optional[str] = None,
    api_key: Optional[str] = None,
    retmax: Optional[int] = None,
    sleep: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, object]:
    """
    Update PMIDs in a payload dictionary (for integration with runner.py).

    Args:
        payload: JSON payload dict with ctx_json
        email: Email for NCBI (optional, uses default from s.py)
        api_key: NCBI API key (optional, uses default from s.py)
        retmax: Max PubMed records to check (optional, uses default from s.py)
        sleep: Delay between API calls (optional, uses default from s.py)
        verbose: Print progress messages

    Returns:
        Modified payload with updated PMIDs
    """
    # Use defaults if not provided
    if email is None:
        email = DEFAULT_EMAIL
    if api_key is None:
        api_key = DEFAULT_API_KEY
    if retmax is None:
        retmax = DEFAULT_RETMAX
    if sleep is None:
        sleep = DEFAULT_SLEEP

    client = PubMedClient(email=email, api_key=api_key, sleep=sleep)

    if verbose:
        print("\n" + "="*80)
        print("UPDATING PMIDs FROM PAPER TITLES")
        print("="*80)

    any_updates = False
    interactor_lists: List[List[Dict[str, object]]] = []
    collect_interactor_lists(payload, interactor_lists)

    total_functions = 0
    updated_functions = 0
    removed_functions = 0

    for interactor_list in interactor_lists:
        idx = len(interactor_list) - 1
        while idx >= 0:
            interactor = interactor_list[idx]
            if not isinstance(interactor, dict):
                idx -= 1
                continue

            functions = interactor.get("functions")
            if not isinstance(functions, list):
                idx -= 1
                continue

            func_idx = len(functions) - 1
            while func_idx >= 0:
                fn_block = functions[func_idx]
                if not isinstance(fn_block, dict):
                    func_idx -= 1
                    continue

                total_functions += 1
                updated, keep = update_function_pmids(fn_block, client, retmax, dry_run=False)

                if updated:
                    any_updates = True
                    updated_functions += 1

                if not keep:
                    func_name = fn_block.get("function", "UNKNOWN")
                    functions.pop(func_idx)
                    if verbose:
                        print(f"  ✗ Deleted function '{func_name}' from interactor '{interactor.get('primary', 'UNKNOWN')}'")
                    removed_functions += 1
                    any_updates = True

                func_idx -= 1

            # Remove interactor if no functions remain
            if isinstance(functions, list) and len(functions) == 0:
                interactor_name = interactor.get("primary") or interactor.get("main") or "UNKNOWN"
                interactor_list.pop(idx)
                if verbose:
                    print(f"  ✗ Removed interactor '{interactor_name}' (no functions with valid PMIDs)")
                any_updates = True

            idx -= 1

    if verbose:
        print(f"\n{'='*80}")
        print("PMID UPDATE SUMMARY")
        print(f"{'='*80}")
        print(f"Total functions processed: {total_functions}")
        print(f"  Updated: {updated_functions}")
        print(f"  Removed: {removed_functions}")
        print(f"  Kept unchanged: {total_functions - updated_functions - removed_functions}")
        print(f"{'='*80}\n")

    return payload


def process_file(path: pathlib.Path, client: PubMedClient, retmax: int, dry_run: bool) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            print(f"[warn] Skipping {path} due to JSON error: {exc}", file=sys.stderr)
            return False

    any_updates = False
    interactor_lists: List[List[Dict[str, object]]] = []
    collect_interactor_lists(payload, interactor_lists)

    for interactor_list in interactor_lists:
        idx = len(interactor_list) - 1
        while idx >= 0:
            interactor = interactor_list[idx]
            if not isinstance(interactor, dict):
                idx -= 1
                continue

            functions = interactor.get("functions")
            if not isinstance(functions, list):
                idx -= 1
                continue

            func_idx = len(functions) - 1
            while func_idx >= 0:
                fn_block = functions[func_idx]
                if not isinstance(fn_block, dict):
                    func_idx -= 1
                    continue

                updated, keep = update_function_pmids(fn_block, client, retmax, dry_run)
                if updated:
                    any_updates = True
                if not keep:
                    func_name = fn_block.get("function", "UNKNOWN")
                    if not dry_run:
                        functions.pop(func_idx)
                    print(f"[info] Deleted function '{func_name}' from interactor '{interactor.get('primary', 'UNKNOWN')}'.")
                    any_updates = True
                func_idx -= 1

            if isinstance(functions, list) and len(functions) == 0:
                interactor_name = interactor.get("primary") or interactor.get("main") or "UNKNOWN"
                if not dry_run:
                    interactor_list.pop(idx)
                print(f"[info] Removed interactor '{interactor_name}' because it no longer has functions.")
                any_updates = True
            idx -= 1

    if any_updates and not dry_run:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return any_updates


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh PMID entries in cache JSON files using PubMed title similarity."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help="Specific JSON files or directories. Defaults to the 'cache' folder in the current working directory.",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=DEFAULT_RETMAX,
        help=f"Maximum PubMed records to inspect per title (default: {DEFAULT_RETMAX}).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Delay between API calls in seconds (default pulled from s.py).",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        help="Email address to pass to NCBI (default pulled from s.py).",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="NCBI API key (default pulled from s.py).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report changes without altering any files.",
    )
    return parser.parse_args(argv)


def discover_targets(paths: List[pathlib.Path]) -> List[pathlib.Path]:
    targets: List[pathlib.Path] = []
    if not paths:
        default = pathlib.Path("cache")
        if default.exists():
            paths = [default]
        else:
            print("[error] No paths provided and default 'cache' directory not found.", file=sys.stderr)
            return []

    for path in paths:
        if path.is_dir():
            targets.extend(sorted(path.rglob("*.json")))
        elif path.suffix.lower() == ".json":
            targets.append(path)
    return targets


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    targets = discover_targets(args.paths)
    if not targets:
        print("[error] No JSON files to process.", file=sys.stderr)
        return 1

    client = PubMedClient(email=args.email, api_key=args.api_key, sleep=args.sleep)
    any_updates = False
    for path in targets:
        print(f"[info] Processing {path}")
        if process_file(path, client, args.retmax, args.dry_run):
            any_updates = True

    if args.dry_run:
        print("[info] Dry run complete.")
    elif not any_updates:
        print("[info] No updates were required.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
