"""
One-shot ingestion script: downloads the latest 10-K for each supported
ticker, chunks the filing, embeds the chunks, and stores them in Chroma.

Run with:
    uv run python -m data.ingest

The script is idempotent — stable chunk IDs mean re-running upserts in place
rather than duplicating, and already-downloaded filings are skipped on disk.

Kept separate from the agent runtime so:
  1. Slow network calls (SEC + embeddings) happen once, offline.
  2. Query-time latency is just retrieval + LLM, no scraping.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_edgar_downloader import Downloader

# Allow running as `python data/ingest.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings  # noqa: E402
from src.tools.rag import get_vectorstore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


# SEC EDGAR requires identifying the requester — these are sent in the User-Agent
SEC_REQUESTER_NAME = "Financial Analyst Demo"
SEC_REQUESTER_EMAIL = "demo@example.com"

FILINGS_DIR = Path(settings.chroma_persist_dir).parent / "data" / "filings"

# Chunk sizing tuned for 10-Ks: large enough to keep risk-factor paragraphs
# intact, small enough that a top-6 retrieval fits comfortably in the context
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_latest_10k(ticker: str) -> None:
    """Fetch the most recent 10-K filing for `ticker` into FILINGS_DIR."""
    logger.info("Downloading latest 10-K for %s ...", ticker)
    dl = Downloader(SEC_REQUESTER_NAME, SEC_REQUESTER_EMAIL, str(FILINGS_DIR))
    # limit=1 fetches only the most recent filing — keeps demo fast
    dl.get("10-K", ticker, limit=1, download_details=True)


def find_filing_files(ticker: str) -> list[Path]:
    """
    Return paths to every downloaded primary-document file for a ticker.

    sec-edgar-downloader writes to:
      FILINGS_DIR/sec-edgar-filings/{ticker}/10-K/{accession}/primary-document.html
    Older versions used full-submission.txt — we accept either.
    """
    base = FILINGS_DIR / "sec-edgar-filings" / ticker / "10-K"
    if not base.exists():
        return []
    files: list[Path] = []
    for accession_dir in base.iterdir():
        if not accession_dir.is_dir():
            continue
        primary = accession_dir / "primary-document.html"
        full = accession_dir / "full-submission.txt"
        if primary.exists():
            files.append(primary)
        elif full.exists():
            files.append(full)
    return files


# ---------------------------------------------------------------------------
# Parse + chunk
# ---------------------------------------------------------------------------

def extract_text(file_path: Path) -> str:
    """
    Strip HTML/SGML and collapse whitespace.

    10-K filings are notoriously messy — tables, inline XBRL, image refs.
    BeautifulSoup with .get_text() removes tags; the regex collapses the
    cascading blank lines that remain.
    """
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def infer_year(file_path: Path) -> str:
    """
    Pull a 4-digit year from the accession folder name (e.g. '0000320193-23-000106' → '23' → '2023').
    Falls back to the file's mtime year if the accession pattern doesn't match.
    """
    accession = file_path.parent.name
    m = re.match(r"\d+-(\d{2})-\d+", accession)
    if m:
        yy = int(m.group(1))
        return str(2000 + yy if yy < 80 else 1900 + yy)
    # Fallback — accession parse failed
    return "unknown"


def chunk_filing(ticker: str, file_path: Path) -> list[dict]:
    """
    Return a list of {id, text, metadata} dicts ready for Chroma.

    Stable IDs ({ticker}_{year}_{idx}) make re-runs idempotent — Chroma's
    upsert semantics replace by ID rather than appending duplicates.
    """
    text = extract_text(file_path)
    year = infer_year(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    return [
        {
            "id": f"{ticker}_{year}_{i}",
            "text": chunk,
            "metadata": {
                "ticker": ticker.upper(),
                "year": year,
                "section": "filing",  # full-section extraction is non-trivial; skip for now
                "source_file": str(file_path.name),
            },
        }
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Embed + store
# ---------------------------------------------------------------------------

def upsert_chunks(records: Iterable[dict], batch_size: int = 50) -> int:
    """
    Upsert chunks into Chroma in batches with retry-on-429.

    The free embedding tier is 100 requests/minute. Sending all ~500 chunks of
    a 10-K in one call blows past that limit, so we split into batches and
    respect the API's `retry in Ns` hint on rate-limit errors.
    """
    records = list(records)
    if not records:
        return 0

    store = get_vectorstore()
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        attempt = 0
        while True:
            try:
                store.add_texts(
                    texts=[r["text"] for r in batch],
                    metadatas=[r["metadata"] for r in batch],
                    ids=[r["id"] for r in batch],
                )
                break
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if not is_rate_limit or attempt >= 5:
                    raise
                # Parse the API's retry_delay hint ("Please retry in 31.3s")
                m = re.search(r"retry in (\d+(?:\.\d+)?)", msg, re.IGNORECASE)
                delay = float(m.group(1)) + 2 if m else 35
                attempt += 1
                logger.warning("  rate-limited (attempt %d) — sleeping %.0fs", attempt, delay)
                time.sleep(delay)
        total += len(batch)
        logger.info("  upserted batch %d/%d (%d/%d total)",
                    (i // batch_size) + 1,
                    (len(records) + batch_size - 1) // batch_size,
                    total, len(records))
        # Small inter-batch breather to keep the rolling window under quota
        time.sleep(1)
    return total


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_ticker(ticker: str) -> None:
    """Download (if needed), chunk, and embed one ticker's most recent 10-K."""
    existing = find_filing_files(ticker)
    if not existing:
        download_latest_10k(ticker)
        existing = find_filing_files(ticker)

    if not existing:
        logger.error("No filing files found for %s after download — skipping.", ticker)
        return

    for file_path in existing:
        logger.info("Chunking %s ...", file_path)
        records = chunk_filing(ticker, file_path)
        logger.info("Embedding %d chunks for %s ...", len(records), ticker)
        written = upsert_chunks(records)
        logger.info("Stored %d chunks for %s.", written, ticker)


def main() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    for ticker in settings.supported_tickers:
        try:
            ingest_ticker(ticker)
        except Exception:
            logger.exception("Ingestion failed for %s — continuing with next.", ticker)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
