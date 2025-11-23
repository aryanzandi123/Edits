#!/usr/bin/env python3
"""Map article titles to their closest matching PubMed identifier (PMID)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DEFAULT_RETMAX = 20
DEFAULT_SLEEP = 0.34  # NCBI recommends <= 3 requests per second without an API key.
DEFAULT_EMAIL = "aryan123zandi@gmail.com"
DEFAULT_API_KEY = "40d004b643f7ecca12f0f72d0b34ab9ef409"


@dataclass
class Match:
    query: str
    pmid: Optional[str]
    matched_title: Optional[str]
    similarity: float

    def as_tsv_row(self) -> str:
        pmid = self.pmid or "NOT_FOUND"
        title = self.matched_title or ""
        score = f"{self.similarity:.4f}"
        return f"{pmid}\t{score}\t{title}\t{self.query}"


class PubMedClient:
    """Thin wrapper around NCBI E-utilities needed for this script."""

    def __init__(self, email: Optional[str], api_key: Optional[str], sleep: float) -> None:
        self._common_params = {"db": "pubmed", "retmode": "json"}
        if email:
            self._common_params["email"] = email
        if api_key:
            self._common_params["api_key"] = api_key
        self._sleep = max(0.0, sleep)

    def _fetch_json(self, endpoint: str, params: Dict[str, str]) -> Dict[str, object]:
        merged = dict(self._common_params)
        merged.update(params)
        url = BASE_URL + endpoint + "?" + urllib.parse.urlencode(merged)

        last_err: Optional[Exception] = None
        for attempt in range(3):
            if attempt:
                time.sleep(self._sleep or 0.1)
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    payload = response.read()
                return json.loads(payload)
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_err = exc
        raise RuntimeError(f"Failed to retrieve data from PubMed: {last_err}")

    def search_ids(self, title: str, retmax: int) -> List[str]:
        params = {
            "term": title,
            "retmax": str(retmax),
            "sort": "relevance",
            "field": "title",
        }
        data = self._fetch_json("esearch.fcgi", params)
        result = data.get("esearchresult", {})
        ids = result.get("idlist", [])
        if not isinstance(ids, list):
            return []
        time.sleep(self._sleep)
        return [str(_id) for _id in ids]

    def fetch_titles(self, pmids: Sequence[str]) -> Dict[str, str]:
        if not pmids:
            return {}
        params = {"id": ",".join(pmids)}
        data = self._fetch_json("esummary.fcgi", params)
        result = data.get("result", {})
        if not isinstance(result, dict):
            return {}
        titles: Dict[str, str] = {}
        for uid in result.get("uids", []):
            payload = result.get(uid, {})
            if not isinstance(payload, dict):
                continue
            title = payload.get("title")
            if isinstance(title, str) and title.strip():
                titles[str(uid)] = title.strip()
        time.sleep(self._sleep)
        return titles


def normalize(text: str) -> str:
    tokens = re.findall(r"\w+", text.lower())
    return " ".join(tokens)


def best_match(query: str, candidates: Dict[str, str]) -> Match:
    if not candidates:
        return Match(query=query, pmid=None, matched_title=None, similarity=0.0)

    norm_query = normalize(query)
    best: Tuple[Optional[str], Optional[str], float] = (None, None, 0.0)
    for pmid, title in candidates.items():
        score = SequenceMatcher(None, norm_query, normalize(title)).ratio()
        if score > best[2]:
            best = (pmid, title, score)
    return Match(query=query, pmid=best[0], matched_title=best[1], similarity=best[2])


def iter_titles(args: argparse.Namespace) -> Iterable[str]:
    seen = False
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            for line in handle:
                title = line.strip()
                if title:
                    seen = True
                    yield title
    for title in args.titles:
        title = title.strip()
        if title:
            seen = True
            yield title
    if not seen and not sys.stdin.isatty():
        for line in sys.stdin:
            title = line.strip()
            if title:
                yield title


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the PubMed ID (PMID) with the most similar title for each provided article title."
    )
    parser.add_argument("titles", nargs="*", help="One or more article titles to query.")
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a UTF-8 text file containing one article title per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path. Defaults to stdout with tab-separated columns: pmid,score,matched_title,query_title.",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        help=f"Email address passed to NCBI (default: {DEFAULT_EMAIL!r}).",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="NCBI API key for higher rate limits (default embedded here; override to use a different key).",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=DEFAULT_RETMAX,
        help=f"Maximum number of PubMed records to fetch per title (default: {DEFAULT_RETMAX}).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Delay in seconds between API calls (default: {DEFAULT_SLEEP}). Use 0 when providing an API key.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    titles = list(iter_titles(args))
    if not titles:
        print("No article titles provided.", file=sys.stderr)
        return 1

    client = PubMedClient(email=args.email, api_key=args.api_key, sleep=args.sleep)
    matches: List[Match] = []
    for title in titles:
        try:
            ids = client.search_ids(title, max(1, args.retmax))
            candidates = client.fetch_titles(ids)
            matches.append(best_match(title, candidates))
        except Exception as exc:  # pylint: disable=broad-except
            matches.append(Match(query=title, pmid=None, matched_title=str(exc), similarity=0.0))

    output_lines = [match.as_tsv_row() for match in matches]
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write("pmid\tscore\tmatched_title\tquery_title\n")
            handle.write("\n".join(output_lines))
            handle.write("\n")
    else:
        print("pmid\tscore\tmatched_title\tquery_title")
        for line in output_lines:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
