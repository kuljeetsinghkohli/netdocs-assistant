from __future__ import annotations

"""
CLI entry point — ``python -m netdocs ingest``.

Usage::

    python -m netdocs ingest [--data-dir PATH] [--reset]

Options:
    --data-dir PATH   Override the raw data directory (default: data/raw).
    --reset           Drop and recreate the ChromaDB collection before ingesting.
    --dry-run         Parse and chunk files without embedding or writing to ChromaDB.
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from netdocs.config import settings

console = Console()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m netdocs",
        description="NetDocs ingestion CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.command == "ingest":
        sys.exit(cmd_ingest(args))


if __name__ == "__main__":
    main()
