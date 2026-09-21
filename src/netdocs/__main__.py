from __future__ import annotations

"""
CLI entry point — ``python -m netdocs``.

Sub-commands::

    python -m netdocs ingest [--data-dir PATH] [--reset] [--dry-run]
    python -m netdocs ask "your question here" [--doc-type TYPE] [--site SITE_ID]
                                               [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD]
                                               [--top-k N]
"""

import argparse
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# Suppress noisy warnings that appear before logging is configured.
# These must be installed before google-auth / urllib3 are first imported.
warnings.filterwarnings("ignore", category=Warning, module=r"urllib3")
warnings.filterwarnings("ignore", message=r".*OpenSSL.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.oauth2")

from rich.console import Console
from rich.table import Table

from netdocs.config import settings

console = Console()
logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for noisy in (
        "httpx",
        "httpcore",
        "sentence_transformers",
        "chromadb",
        "urllib3",
        "google.auth",
        "google.auth.transport",
        "google_auth_httplib2",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Suppress urllib3 NotOpenSSLWarning and google-auth FutureWarning at the
    # warnings module level so they never reach stderr.
    import warnings
    warnings.filterwarnings("ignore", category=Warning, module=r"urllib3")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.auth")
    warnings.filterwarnings("ignore", message=r".*OpenSSL.*", category=Warning)
    warnings.filterwarnings("ignore", message=r".*urllib3.*", category=Warning)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m netdocs",
        description="NetDocs CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    ingest = sub.add_parser("ingest", help="Ingest documents into ChromaDB")
    ingest.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to raw document directory (default: settings.data_dir)",
    )
    ingest.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the ChromaDB collection before ingesting",
    )
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; do not embed or write to ChromaDB",
    )
    ingest.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )

    # --- ask ---
    ask_p = sub.add_parser("ask", help="Ask a question against the document corpus")
    ask_p.add_argument("question", help="The question to answer")
    ask_p.add_argument(
        "--doc-type",
        dest="doc_type",
        nargs="+",
        choices=["design_doc", "runbook", "ticket", "config"],
        default=None,
        help="Restrict retrieval to one or more document types",
    )
    ask_p.add_argument(
        "--site",
        dest="site_id",
        default=None,
        help="Restrict retrieval to a specific site ID (e.g. LON-DC01)",
    )
    ask_p.add_argument(
        "--date-from",
        dest="date_from",
        default=None,
        help="Filter tickets by change_date >= YYYY-MM-DD",
    )
    ask_p.add_argument(
        "--date-to",
        dest="date_to",
        default=None,
        help="Filter tickets by change_date <= YYYY-MM-DD",
    )
    ask_p.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Number of context chunks to retrieve (default: 5)",
    )
    ask_p.add_argument(
        "--no-rerank",
        dest="no_rerank",
        action="store_true",
        help="Skip cross-encoder reranking (faster but lower quality)",
    )

    return parser.parse_args(argv)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Execute the ingest command.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    _configure_logging(settings.log_level)

    from netdocs.ingestion.loader import load_all
    from netdocs.embeddings.embedder import get_embedder
    from netdocs.retriever.vector_store import VectorStore

    data_dir = args.data_dir or settings.data_dir
    console.rule("[bold cyan]NetDocs Ingestion Pipeline")
    console.print(f"  Data dir  : [green]{data_dir}[/]")
    console.print(f"  Chroma    : [green]{settings.chroma_path}[/]")
    console.print(f"  Collection: [green]{settings.chroma_collection}[/]")
    console.print(f"  Embedder  : [green]{settings.embed_backend} / {settings.embed_model}[/]")
    console.print()

    # --- 1. Load and chunk ---
    t0 = time.perf_counter()
    console.print("[bold]Step 1:[/] Loading and chunking documents…")
    try:
        chunks = load_all(data_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]ERROR:[/] {exc}")
        return 1

    t_load = time.perf_counter() - t0
    console.print(
        f"  → [green]{len(chunks)}[/] chunks from [cyan]{data_dir}[/] "
        f"in [yellow]{t_load:.1f}s[/]\n"
    )

    if args.dry_run:
        _print_statistics(chunks)
        console.print("[yellow]Dry-run mode: skipping embedding and upsert.[/]")
        return 0

    # --- 2. Embed ---
    console.print("[bold]Step 2:[/] Embedding chunks…")
    t1 = time.perf_counter()
    embedder = get_embedder()
    texts = [c.text for c in chunks]

    all_embeddings: list[list[float]] = []
    batch_size = args.batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vecs = embedder.embed(batch)
        all_embeddings.extend(batch_vecs)
        console.print(
            f"  Embedded [{start + len(batch)}/{len(texts)}]",
            end="\r",
        )

    t_embed = time.perf_counter() - t1
    console.print(
        f"\n  → [green]{len(all_embeddings)}[/] vectors "
        f"(dim={embedder.dimension}) in [yellow]{t_embed:.1f}s[/]\n"
    )

    # --- 3. Upsert into ChromaDB ---
    console.print("[bold]Step 3:[/] Upserting into ChromaDB…")
    t2 = time.perf_counter()
    store = VectorStore()

    if args.reset:
        console.print("  [yellow]--reset flag set: dropping existing collection…[/]")
        store._client.delete_collection(settings.chroma_collection)
        store = VectorStore()  # recreate

    store.upsert_chunks(chunks, all_embeddings)
    t_upsert = time.perf_counter() - t2
    total_count = store.count()

    console.print(
        f"  → ChromaDB collection now contains [green]{total_count}[/] documents "
        f"in [yellow]{t_upsert:.1f}s[/]\n"
    )

    # --- 4. Statistics ---
    _print_statistics(chunks)
    total_time = time.perf_counter() - t0
    console.print(f"\n[bold green]✓ Ingestion complete in {total_time:.1f}s[/]")
    return 0


def _print_statistics(chunks) -> None:
    """Print a summary table of chunk statistics."""
    from netdocs.ingestion.models import Chunk

    by_type: Counter = Counter(c.doc_type for c in chunks)
    token_by_type: dict[str, list[int]] = {}
    for c in chunks:
        token_by_type.setdefault(c.doc_type, []).append(len(c.text.split()))

    table = Table(title="Chunk Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Doc Type", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Min Tokens", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Max Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")

    grand_chunks = 0
    grand_tokens = 0

    for doc_type in sorted(by_type):
        counts = token_by_type[doc_type]
        n = len(counts)
        total = sum(counts)
        table.add_row(
            doc_type,
            str(n),
            str(min(counts)),
            str(round(total / n)),
            str(max(counts)),
            str(total),
        )
        grand_chunks += n
        grand_tokens += total

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{grand_chunks}[/]",
        "",
        "",
        "",
        f"[bold]{grand_tokens}[/]",
    )

    console.print(table)


def cmd_ask(args: argparse.Namespace) -> int:
    """Execute the ask command.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    _configure_logging(settings.log_level)

    from netdocs.embeddings.embedder import get_embedder
    from netdocs.retriever.vector_store import VectorStore
    from netdocs.retriever.bm25_index import BM25Index
    from netdocs.retriever.hybrid import HybridRetriever
    from netdocs.retriever.reranker import get_reranker
    from netdocs.llm.client import get_llm_client
    from netdocs.llm.generator import generate_answer

    from netdocs.llm.client import _resolve_provider

    console.rule("[bold cyan]NetDocs Ask")
    console.print(f"  Question : [bold]{args.question}[/]")
    effective_provider = _resolve_provider()
    console.print(f"  Provider : [green]{effective_provider} / {settings.llm_model}[/]")

    # Build filters
    filters: dict = {}
    if args.doc_type:
        filters["doc_type"] = args.doc_type
    if args.site_id:
        filters["site_id"] = args.site_id
    if args.date_from:
        filters["date_from"] = args.date_from
    if args.date_to:
        filters["date_to"] = args.date_to

    # Initialise pipeline
    try:
        t0 = time.perf_counter()
        embedder = get_embedder()
        store = VectorStore()
        bm25 = BM25Index.load_or_build(store)
        retriever = HybridRetriever(store, bm25, embedder)
        top_k = args.top_k or settings.retrieval_top_k
        candidates = retriever.retrieve(
            args.question,
            top_k=top_k * 4,
            candidate_k=settings.retrieval_candidate_k,
            filters=filters or None,
        )

        if not args.no_rerank:
            reranker = get_reranker()
            top_chunks = reranker.rerank(args.question, candidates, top_k=top_k)
        else:
            top_chunks = candidates[:top_k]

        llm = get_llm_client()
        result = generate_answer(args.question, top_chunks, llm)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        console.print(f"[red]ERROR:[/] {exc}")
        logger.exception("ask command failed")
        return 1

    # --- Output ---
    console.print()
    if result.refused:
        console.print(f"[yellow]⚠ Refused:[/] {result.answer}")
    else:
        console.rule("[bold green]Answer")
        console.print(result.answer)
        if result.citations:
            console.print()
            console.rule("[dim]Sources")
            for cit in result.citations:
                console.print(
                    f"  [cyan]{cit.doc_id}[/]  "
                    f"[dim]{cit.doc_type}[/]  "
                    f"§ {cit.section or '—'}  "
                    f"[dim]{cit.source_file}[/]"
                )
    console.print(f"\n[dim]confidence={result.confidence:.4f}  elapsed={elapsed:.1f}s[/]")
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.command == "ingest":
        sys.exit(cmd_ingest(args))
    elif args.command == "ask":
        sys.exit(cmd_ask(args))


if __name__ == "__main__":
    main()
