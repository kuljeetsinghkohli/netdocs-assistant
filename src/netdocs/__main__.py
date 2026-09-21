from __future__ import annotations

"""
CLI entry point — ``python -m netdocs``.

Sub-commands::

    python -m netdocs ingest [--data-dir PATH] [--reset] [--dry-run]
    python -m netdocs ask "your question here" [--doc-type TYPE] [--site SITE_ID]
                                               [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD]
                                               [--top-k N]
    python -m netdocs eval [--top-k N] [--out PATH] [--threshold FLOAT]
    python -m netdocs agent "your question here" [--max-steps N] [--approve]
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

    # --- agent ---
    agent_p = sub.add_parser(
        "agent",
        help="Answer a question using the agent tool-calling loop",
    )
    agent_p.add_argument("question", help="The question for the agent to answer")
    agent_p.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=8,
        help="Maximum tool-call iterations before the agent aborts (default: 8)",
    )
    agent_p.add_argument(
        "--approve",
        dest="require_approval",
        action="store_true",
        help=(
            "Gate state-changing tools behind human approval prompts. "
            "When omitted, state-changing tools run without prompting."
        ),
    )

    # --- eval ---
    eval_p = sub.add_parser(
        "eval",
        help="Run retrieval-only evaluation on the golden Q&A set (no LLM calls)",
    )
    eval_p.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    eval_p.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=None,
        help=(
            "Confidence threshold for refusal accuracy (default: "
            "settings.retrieval_confidence_threshold)"
        ),
    )
    eval_p.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=None,
        help="Path for the Markdown report (default: reports/eval_report.md)",
    )

    return parser.parse_args(argv)  # type: ignore[return-value]


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


def _get_active_model_name() -> str:
    """Return the real model name for the active provider.

    Resolves which provider is actually being used (accounting for auto-fallback)
    and returns the concrete model string, not the static ``settings.llm_model``
    which may still read "gpt-4o-mini" when the provider fell back to Gemini.
    """
    from netdocs.llm.client import _resolve_provider

    provider = _resolve_provider()
    if provider == "gemini":
        return settings.gemini_model or settings.llm_model
    if provider in ("openai", "ollama"):
        return settings.llm_model
    return provider  # "extractive" / "fake" — just show the provider name


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
    from netdocs.llm.client import get_llm_client, _resolve_provider
    from netdocs.llm.generator import generate_answer, REFUSAL_LOW_CONFIDENCE, REFUSAL_LLM_DECLINED, REFUSAL_EMPTY_GENERATION

    console.rule("[bold cyan]NetDocs Ask")
    console.print(f"  Question : [bold]{args.question}[/]")
    effective_provider = _resolve_provider()
    active_model = _get_active_model_name()
    console.print(f"  Provider : [green]{effective_provider} / {active_model}[/]")

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
            # Apply doc-type boost after reranking to correct systematic mis-ordering
            from netdocs.retriever.hybrid import apply_doc_type_boost
            top_chunks = apply_doc_type_boost(top_chunks, args.question)
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
        reason_labels = {
            REFUSAL_LOW_CONFIDENCE: "Low retrieval confidence — no sufficiently relevant chunks found",
            REFUSAL_LLM_DECLINED: "LLM declined to answer — context passages did not contain enough information",
            REFUSAL_EMPTY_GENERATION: "Empty generation — LLM returned no text",
        }
        reason_label = reason_labels.get(
            result.refusal_reason,
            f"Refused ({result.refusal_reason or 'unknown reason'})",
        )
        console.print(f"[yellow]⚠ {reason_label}[/]")
        console.print(f"  {result.answer}")
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


def cmd_eval(args: argparse.Namespace) -> int:
    """Execute the eval command — retrieval-only, no LLM calls.

    Returns:
        Exit code (0 = all answerable passed, 1 = some failed).
    """
    _configure_logging(settings.log_level)

    from netdocs.eval.harness import run_eval
    from netdocs.eval.reporter import write_report

    console.rule("[bold cyan]NetDocs Retrieval Evaluation")
    console.print(f"  Top-K     : [green]{args.top_k}[/]")
    if args.threshold is not None:
        console.print(f"  Threshold : [green]{args.threshold}[/]")
    console.print()

    t0 = time.perf_counter()
    try:
        report = run_eval(
            top_k=args.top_k,
            confidence_threshold=args.threshold,
        )
    except Exception as exc:
        console.print(f"[red]ERROR:[/] {exc}")
        logger.exception("eval command failed")
        return 1

    elapsed = time.perf_counter() - t0

    # Print summary
    console.rule("[bold green]Eval Results")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Answerable questions", str(report.n_answerable))
    table.add_row("Unanswerable questions", str(report.n_unanswerable))
    table.add_row("Hit@1", f"{report.hit_at_1:.1%}")
    table.add_row("Hit@3", f"{report.hit_at_3:.1%}")
    table.add_row("Hit@5", f"{report.hit_at_5:.1%}")
    table.add_row("MRR", f"{report.mrr:.4f}")
    table.add_row("Refusal accuracy", f"{report.refusal_accuracy:.1%}")
    console.print(table)

    # Per-question failures
    failures = [
        r for r in report.results
        if (not r.unanswerable and not (r.rank and r.rank <= report.top_k))
        or (r.unanswerable and not r.refusal_correct)
    ]
    if failures:
        console.print(f"\n[yellow]Failures ({len(failures)}):[/]")
        for r in failures:
            if r.unanswerable:
                top_score = r.retrieved_scores[0] if r.retrieved_scores else 0.0
                console.print(
                    f"  [red]{r.qid}[/] (unanswerable not refused): "
                    f"top_score={top_score:.3f}  {r.question[:60]}"
                )
            else:
                console.print(
                    f"  [red]{r.qid}[/] (miss rank={r.rank}): "
                    f"expected={r.expected_sources}  {r.question[:60]}"
                )
    else:
        console.print("\n[bold green]All questions passed![/]")

    # Write report
    out_path = write_report(report, args.out)
    console.print(f"\n[dim]Report written to: {out_path}[/]")
    console.print(f"[dim]Total eval time: {elapsed:.1f}s[/]")

    # Exit 1 if any answerable question is a miss (regression guard)
    has_misses = any(
        not r.unanswerable and not (r.rank and r.rank <= report.top_k)
        for r in report.results
    )
    return 1 if has_misses else 0


def cmd_agent(args: argparse.Namespace) -> int:
    """Execute the agent command.

    Returns:
        Exit code (0 = success, 1 = failure / aborted).
    """
    _configure_logging(settings.log_level)

    from netdocs.agent.loop import run_agent
    from netdocs.llm.client import get_llm_client

    console.rule("[bold cyan]NetDocs Agent")
    console.print(f"  Question   : [bold]{args.question}[/]")
    console.print(f"  Max steps  : [green]{args.max_steps}[/]")
    console.print(f"  Approval   : [green]{'yes' if args.require_approval else 'no (auto)'}[/]")
    console.print()

    t0 = time.perf_counter()
    try:
        llm = get_llm_client()
        result = run_agent(
            args.question,
            llm_client=llm,
            max_steps=args.max_steps,
            require_approval=args.require_approval,
        )
    except Exception as exc:
        console.print(f"[red]ERROR:[/] {exc}")
        logger.exception("agent command failed")
        return 1

    elapsed = time.perf_counter() - t0

    # --- Print step trace ---
    if result.steps:
        console.rule("[dim]Agent Steps")
        for i, step in enumerate(result.steps, 1):
            stype = getattr(step, "type", "?")
            if stype == "tool_call":
                console.print(
                    f"  [cyan]Step {i}:[/] tool_call → [green]{step.tool}[/]"  # type: ignore[union-attr]
                    f"  ({step.duration_ms:.0f}ms)"  # type: ignore[union-attr]
                )
            elif stype == "error":
                console.print(
                    f"  [red]Step {i}:[/] error in [yellow]{step.tool}[/]"  # type: ignore[union-attr]
                    f" ({step.error_kind}): {step.error}"  # type: ignore[union-attr]
                )
            elif stype == "final_answer":
                console.print(f"  [dim]Step {i}:[/] final_answer")

    # --- Print final answer ---
    console.print()
    if result.aborted:
        console.print(f"[yellow]⚠ Agent aborted: {result.abort_reason}[/]")
    if result.degraded:
        console.print("[yellow]⚠ LLM degraded (fell back to extractive provider)[/]")
    console.rule("[bold green]Agent Answer")
    console.print(result.final_answer or "[dim](no answer generated)[/]")
    console.print(f"\n[dim]steps={len(result.steps)}  elapsed={elapsed:.1f}s[/]")
    return 1 if result.aborted else 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.command == "ingest":
        sys.exit(cmd_ingest(args))
    elif args.command == "ask":
        sys.exit(cmd_ask(args))
    elif args.command == "eval":
        sys.exit(cmd_eval(args))
    elif args.command == "agent":
        sys.exit(cmd_agent(args))


if __name__ == "__main__":
    main()
