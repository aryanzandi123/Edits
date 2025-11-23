#!/usr/bin/env python3
"""
PMID Extractor - Non-AI based tool to extract correct PMIDs from PubMed
Uses NCBI E-utilities API to reliably fetch PMIDs given DOI or paper title
"""

import os
import random
import re
import time
import urllib.parse
import urllib.request
import urllib.error
import ssl
import xml.etree.ElementTree as ET
from functools import wraps
from typing import Optional, Dict, Any, Callable

try:
    import certifi  # type: ignore
except ImportError:  # pragma: no cover - best effort fallback
    certifi = None


_SSL_CONTEXT = None

# Get NCBI API key from environment (optional but recommended)
# Register for free API key at https://www.ncbi.nlm.nih.gov/account/
# This increases rate limit from 3/sec to 10/sec
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")


def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 32.0,
    exponential_base: float = 2.0
) -> Callable:
    """
    Decorator that retries a function with exponential backoff on rate limit errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation

    Returns:
        Decorated function that retries on HTTP 429/503 errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except urllib.error.HTTPError as e:
                    # Retry on rate limit (429) or service unavailable (503)
                    if e.code in (429, 503):
                        if retries >= max_retries:
                            print(f"    [!] Max retries ({max_retries}) exceeded for {func.__name__}")
                            raise

                        # Calculate delay with exponential backoff and jitter
                        delay = min(base_delay * (exponential_base ** retries), max_delay)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter

                        retries += 1
                        print(f"    [!] Rate limit hit (HTTP {e.code}). Retry {retries}/{max_retries} after {total_delay:.1f}s...")
                        time.sleep(total_delay)
                    else:
                        # Re-raise non-retryable HTTP errors
                        raise
                except Exception as e:
                    # Re-raise all other exceptions
                    raise

            # Should never reach here
            raise Exception(f"Failed after {max_retries} retries")

        return wrapper
    return decorator


def _get_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that prefers certifi's CA bundle when available."""
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            if certifi is not None:
                _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
            else:
                _SSL_CONTEXT = ssl.create_default_context()
        except Exception:
            # Fall back to Python's default context if custom init fails
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


def _urlopen(url: str, *, timeout: float = 10):
    """
    Wrapper around urllib.request.urlopen that injects our SSL context while preserving
    compatibility with older Python versions that may not accept the context argument.
    """
    context = _get_ssl_context()
    try:
        return urllib.request.urlopen(url, timeout=timeout, context=context)
    except TypeError:
        return urllib.request.urlopen(url, timeout=timeout)


def clean_doi(doi: str) -> str:
    """Remove common prefixes from DOI."""
    if not doi:
        return ""
    doi_clean = doi.replace('doi:', '').replace('DOI:', '').strip()
    doi_clean = doi_clean.replace('https://doi.org/', '').replace('http://doi.org/', '')
    return doi_clean


@retry_with_backoff(max_retries=5, base_delay=2.0)
def extract_pmid_from_doi(doi: str, email: str = "research@example.com") -> Optional[str]:
    """
    Extract PMID from DOI using PubMed API.

    Args:
        doi: DOI string (e.g., "10.1016/j.cell.2014.08.017")
        email: Email for NCBI API (required by their policy)

    Returns:
        PMID as string, or None if not found
    """
    doi = clean_doi(doi)
    if not doi:
        return None

    # Use PubMed ESearch API to find PMID from DOI
    # Format: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=DOI
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': f'{doi}[DOI]',
        'retmode': 'xml',
        'email': email
    }

    # Add API key if available
    if NCBI_API_KEY:
        params['api_key'] = NCBI_API_KEY

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with _urlopen(url, timeout=10) as response:
            xml_data = response.read().decode('utf-8')

        # Parse XML
        root = ET.fromstring(xml_data)
        id_list = root.find('IdList')

        if id_list is not None:
            pmid_elem = id_list.find('Id')
            if pmid_elem is not None and pmid_elem.text:
                return pmid_elem.text.strip()

        return None

    except urllib.error.HTTPError as e:
        # Re-raise HTTP errors for retry decorator to handle
        if e.code in (429, 503):
            raise
        # Use ASCII-safe warning for Windows compatibility
        print(f"    [!] Error fetching PMID for DOI {doi}: HTTP Error {e.code}")
        return None
    except Exception as e:
        # Use ASCII-safe warning for Windows compatibility
        print(f"    [!] Error fetching PMID for DOI {doi}: {e}")
        return None


@retry_with_backoff(max_retries=5, base_delay=2.0)
def extract_pmid_from_title(title: str, email: str = "research@example.com") -> Optional[str]:
    """
    Extract PMID from paper title using PubMed API.

    Args:
        title: Paper title
        email: Email for NCBI API

    Returns:
        PMID as string, or None if not found
    """
    if not title or len(title) < 10:
        return None

    # Use PubMed ESearch API to search by title
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': f'{title}[Title]',
        'retmode': 'xml',
        'retmax': 1,  # Only get top result
        'email': email
    }

    # Add API key if available
    if NCBI_API_KEY:
        params['api_key'] = NCBI_API_KEY

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with _urlopen(url, timeout=10) as response:
            xml_data = response.read().decode('utf-8')

        # Parse XML
        root = ET.fromstring(xml_data)
        id_list = root.find('IdList')

        if id_list is not None:
            pmid_elem = id_list.find('Id')
            if pmid_elem is not None and pmid_elem.text:
                return pmid_elem.text.strip()

        return None

    except urllib.error.HTTPError as e:
        # Re-raise HTTP errors for retry decorator to handle
        if e.code in (429, 503):
            raise
        print(f"    [!] Error fetching PMID for title: HTTP Error {e.code}")
        return None
    except Exception as e:
        print(f"    [!] Error fetching PMID for title: {e}")
        return None


@retry_with_backoff(max_retries=5, base_delay=2.0)
def get_paper_metadata(pmid: str, email: str = "research@example.com") -> Optional[Dict[str, Any]]:
    """
    Fetch complete paper metadata from PMID using PubMed API.

    Args:
        pmid: PubMed ID
        email: Email for NCBI API

    Returns:
        Dict with paper metadata (title, authors, journal, year, doi), or None
    """
    if not pmid:
        return None

    # Use PubMed ESummary API to get metadata
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml',
        'email': email
    }

    # Add API key if available
    if NCBI_API_KEY:
        params['api_key'] = NCBI_API_KEY

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with _urlopen(url, timeout=10) as response:
            xml_data = response.read().decode('utf-8')

        # Parse XML
        root = ET.fromstring(xml_data)
        doc_sum = root.find('.//DocumentSummary')

        if doc_sum is None:
            return None

        metadata = {}

        # Extract title
        title_elem = doc_sum.find('.//Item[@Name="Title"]')
        if title_elem is not None and title_elem.text:
            metadata['title'] = title_elem.text.strip()

        # Extract authors
        authors = []
        for author_elem in doc_sum.findall('.//Item[@Name="Author"]'):
            if author_elem.text:
                authors.append(author_elem.text.strip())
        if authors:
            metadata['authors'] = ', '.join(authors[:3])  # First 3 authors

        # Extract journal
        source_elem = doc_sum.find('.//Item[@Name="Source"]')
        if source_elem is not None and source_elem.text:
            metadata['journal'] = source_elem.text.strip()

        # Extract year
        pub_date_elem = doc_sum.find('.//Item[@Name="PubDate"]')
        if pub_date_elem is not None and pub_date_elem.text:
            # Extract year from date string
            year_match = re.search(r'\b(19|20)\d{2}\b', pub_date_elem.text)
            if year_match:
                metadata['year'] = int(year_match.group(0))

        # Extract DOI (from ArticleIds)
        for id_elem in doc_sum.findall('.//Item[@Name="ArticleIds"]/Item'):
            if id_elem.get('Name') == 'doi' and id_elem.text:
                metadata['doi'] = clean_doi(id_elem.text.strip())
                break

        metadata['pmid'] = pmid

        return metadata if metadata else None

    except urllib.error.HTTPError as e:
        # Re-raise HTTP errors for retry decorator to handle
        if e.code in (429, 503):
            raise
        print(f"    [!] Error fetching metadata for PMID {pmid}: HTTP Error {e.code}")
        return None
    except Exception as e:
        print(f"    [!] Error fetching metadata for PMID {pmid}: {e}")
        return None


def extract_pmid_smart(paper_info: Dict[str, Any], email: str = "research@example.com") -> Optional[str]:
    """
    Smart PMID extraction that tries multiple strategies.

    Args:
        paper_info: Dict with any of: doi, paper_title, title, pmid
        email: Email for NCBI API

    Returns:
        PMID as string, or None if not found
    """
    # Strategy 1: If PMID already exists and looks valid (8 digits), verify it
    existing_pmid = paper_info.get('pmid', '')
    if existing_pmid and re.match(r'^\d{7,8}$', str(existing_pmid)):
        # Verify it exists
        metadata = get_paper_metadata(existing_pmid, email)
        if metadata:
            return existing_pmid

    # Strategy 2: Try DOI first (most reliable)
    doi = paper_info.get('doi', '')
    if doi:
        pmid = extract_pmid_from_doi(doi, email)
        if pmid:
            return pmid
        # Rate limit: NCBI allows 3 req/sec (no key) or 10 req/sec (with key)
        # Add jitter to prevent synchronized bursts
        delay = 0.5 if not NCBI_API_KEY else 0.15
        time.sleep(delay + random.uniform(0, 0.1))

    # Strategy 3: Try title
    title = paper_info.get('paper_title') or paper_info.get('title', '')
    if title:
        pmid = extract_pmid_from_title(title, email)
        if pmid:
            return pmid
        delay = 0.5 if not NCBI_API_KEY else 0.15
        time.sleep(delay + random.uniform(0, 0.1))

    return None


def verify_and_enrich_evidence(evidence_list: list, email: str = "research@example.com") -> list:
    """
    Verify and enrich a list of evidence entries with correct PMIDs and metadata.

    Args:
        evidence_list: List of evidence dicts from pipeline JSON
        email: Email for NCBI API

    Returns:
        Updated evidence list with corrected PMIDs and enriched metadata
    """
    enriched = []

    for evidence in evidence_list:
        # Try to extract correct PMID
        correct_pmid = extract_pmid_smart(evidence, email)

        if correct_pmid:
            # Fetch complete metadata
            metadata = get_paper_metadata(correct_pmid, email)

            if metadata:
                # Update evidence with verified data
                evidence['pmid'] = correct_pmid
                if 'doi' in metadata:
                    evidence['doi'] = metadata['doi']
                if 'title' in metadata and 'paper_title' not in evidence:
                    evidence['paper_title'] = metadata['title']
                if 'authors' in metadata:
                    evidence['authors'] = metadata['authors']
                if 'journal' in metadata:
                    evidence['journal'] = metadata['journal']
                if 'year' in metadata:
                    evidence['year'] = metadata['year']

                evidence['pmid_verified'] = True
            else:
                evidence['pmid'] = correct_pmid
                evidence['pmid_verified'] = False
        else:
            evidence['pmid_verified'] = False
            evidence['pmid_error'] = 'Could not extract PMID from available information'

        enriched.append(evidence)
        # Rate limit with jitter
        delay = 0.5 if not NCBI_API_KEY else 0.15
        time.sleep(delay + random.uniform(0, 0.1))

    return enriched


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python pmid_extractor.py doi <DOI>")
        print("  python pmid_extractor.py title <TITLE>")
        print("  python pmid_extractor.py pmid <PMID>  (to fetch metadata)")
        sys.exit(1)

    mode = sys.argv[1].lower()
    value = ' '.join(sys.argv[2:])

    if mode == 'doi':
        pmid = extract_pmid_from_doi(value)
        if pmid:
            print(f"PMID: {pmid}")
            metadata = get_paper_metadata(pmid)
            if metadata:
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"Authors: {metadata.get('authors', 'N/A')}")
                print(f"Journal: {metadata.get('journal', 'N/A')}")
                print(f"Year: {metadata.get('year', 'N/A')}")
        else:
            print("PMID not found")

    elif mode == 'title':
        pmid = extract_pmid_from_title(value)
        if pmid:
            print(f"PMID: {pmid}")
            metadata = get_paper_metadata(pmid)
            if metadata:
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"DOI: {metadata.get('doi', 'N/A')}")
        else:
            print("PMID not found")

    elif mode == 'pmid':
        metadata = get_paper_metadata(value)
        if metadata:
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Authors: {metadata.get('authors', 'N/A')}")
            print(f"Journal: {metadata.get('journal', 'N/A')}")
            print(f"Year: {metadata.get('year', 'N/A')}")
            print(f"DOI: {metadata.get('doi', 'N/A')}")
        else:
            print("Metadata not found")

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
