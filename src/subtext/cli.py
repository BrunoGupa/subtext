"""`subtext` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_rows(rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        print("(no rows)")
        return
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))[:60]) for r in rows)) for c in columns}
    print("  ".join(str(c).ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, ""))[:60].ljust(widths[c]) for c in columns))


def cmd_init_db(args: argparse.Namespace) -> int:
    from .db import init_db

    init_db()
    print("schema ready")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download the third-party corpora. Nothing is redistributed by this repo."""
    from .ingest.fetch import all_datasets, download, human_size

    catalogue = all_datasets(args.lang_pair)
    if args.what == "all":
        wanted = list(catalogue.values())
    else:
        wanted = [d for k, d in catalogue.items() if k.startswith(args.what)]
    if not wanted:
        print(f"nothing matches {args.what!r}; known: {', '.join(catalogue)}", file=sys.stderr)
        return 2

    pending = [d for d in wanted if not d.destination.exists() or args.force]
    total = sum(d.approx_bytes for d in pending)

    print("These datasets are NOT redistributed by this repository. Downloading from source:\n")
    for dataset in wanted:
        state = "present" if dataset.destination.exists() and not args.force else "will download"
        print(f"  {dataset.key:24} {dataset.approx_size:>9}  {state}")
        print(f"  {'':24} {dataset.description}")
    if not pending:
        print("\nEverything is already present. Use --force to re-download.")
        return 0
    print(f"\ntotal to download: {human_size(total)} -> data/raw/ (gitignored)")
    print("\nBy using these you accept their terms; see CORPUS.md.")
    print("  OpenSubtitles/OPUS: cite Lison & Tiedemann 2016 and link opensubtitles.org")
    print("  IMDb: personal and non-commercial use only, attribution required\n")

    if not args.yes:
        reply = input("proceed? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("aborted")
            return 1

    for dataset in pending:
        print(f"\n{dataset.key} ...")

        def progress(seen: int, expected: int, _key: str = dataset.key) -> None:
            if expected:
                pct = 100.0 * seen / expected
                print(f"\r  {human_size(seen)} / {human_size(expected)}  ({pct:.1f}%)", end="")

        download(dataset, force=args.force, on_progress=progress)
        print(f"\r  saved {dataset.destination.relative_to(Path.cwd()) if dataset.destination.is_relative_to(Path.cwd()) else dataset.destination}          ")
    print("\ndone")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    from .ingest.load import load_corpus

    count = load_corpus(
        args.source,
        path=Path(args.path) if args.path else None,
        title=args.title,
        title_id=args.title_id,
    )
    print(f"loaded {count} lines from source={args.source}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .ingest.load import build_chunks

    count = build_chunks(
        strategy=args.strategy,
        window_size=args.window_size,
        stride=args.stride,
        show_progress=True,
    )
    label = args.strategy if args.strategy == "line" else f"{args.strategy}{args.window_size}"
    print(f"embedded {count} chunks (strategy={label})")
    return 0


def cmd_embed_schema(args: argparse.Namespace) -> int:
    from .schema_retrieval import build_schema_docs

    print(f"embedded {build_schema_docs()} schema column descriptions")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .agent import clickhouse_mcp
    from .config import settings
    from .ingest.load import loaded_strategies
    from .retrieval import corpus_stats

    s = settings()
    stats = corpus_stats()
    print(f"clickhouse   {s.ch_host}:{s.ch_port}/{s.ch_database}")
    print(f"embeddings   {s.embedding_model}")
    print(f"gemini       {s.gemini_model} (key {'set' if s.has_gemini_key else 'NOT set'})")
    print(f"mcp          mcp-clickhouse {'on PATH' if clickhouse_mcp.is_available() else 'NOT FOUND - run uv sync'}")
    print(f"corpus       {stats['lines']} lines · {stats['titles']} titles · "
          f"{stats['characters']} characters · {stats['seasons']} seasons")
    chunks = loaded_strategies()
    if not chunks:
        print("chunks       none - run: subtext embed")
    for strategy, window, count in chunks:
        print(f"chunks       {strategy} window={window}: {count}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .retrieval import search

    hits = search(
        args.question,
        k=args.k,
        strategy=args.strategy,
        window_size=args.window_size,
        character=args.character,
        involving=args.involving,
        season=args.season,
    )
    _print_rows(
        [
            {
                "line_id": h.line_id,
                "s/e": f"S{h.season:02d}E{h.episode:02d}",
                "character": h.character,
                "timecode": h.timecode,
                "distance": round(h.distance, 4),
                "text": h.text,
            }
            for h in hits
        ]
    )
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from .retrieval import hybrid_aggregate

    columns = tuple(c.strip() for c in args.group_by.split(","))
    rows, sql = hybrid_aggregate(
        args.question,
        group_by=columns,
        k=args.k,
        strategy=args.strategy,
        window_size=args.window_size,
        character=args.character,
        involving=args.involving,
        max_distance=args.max_distance,
    )
    _print_rows(
        [
            {**{c: r[c] for c in columns}, "matches": r["matches"], "best_distance": r["best_distance"]}
            for r in rows
        ]
    )
    print(f"\ntotal: {sum(int(r['matches']) for r in rows)}")
    if args.show_sql:
        print(f"\nSQL:\n{sql}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from .schema_retrieval import schema_context

    context = schema_context(args.question, k=args.k)
    print(f"retrieved {context['columns_retrieved']} of {context['columns_available']} columns "
          f"across {', '.join(context['tables'])}\n")
    print(context["prompt_block"])
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .config import settings

    if not settings().has_gemini_key:
        print(
            "GOOGLE_API_KEY is not set. Add one to .env (free tier at "
            "https://aistudio.google.com/apikey), or use `subtext aggregate` "
            "for the retrieval path without the model.",
            file=sys.stderr,
        )
        return 2

    from .agent import ask

    answer = ask(args.question, model=args.model, use_mcp=not args.no_mcp)
    print(answer.text)
    if args.trace:
        print("\n--- trace ---")
        for call in answer.tool_calls:
            print(f"  {call['name']}({json.dumps(call['args'], default=str)[:160]})")
        print(f"  retrieved line_ids: {len(answer.retrieved_line_ids)}")
        print(f"  self-verification: {answer.verified}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web import serve

    print(f"subtext on http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import run_eval

    run_eval(
        golden_path=Path(args.golden) if args.golden else None,
        ks=[int(k) for k in args.ks.split(",")],
        strategy=args.strategy,
        window_size=args.window_size,
        with_agent=args.with_agent,
        limit=args.limit,
        output=Path(args.output) if args.output else None,
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .evaluation import run_sweep

    run_sweep(
        golden_path=Path(args.golden) if args.golden else None,
        ks=[int(k) for k in args.ks.split(",")],
        strategies=[s.strip() for s in args.strategies.split(",")],
        output=Path(args.output) if args.output else None,
    )
    return 0


def cmd_build_mx(args: argparse.Namespace) -> int:
    """Rebuild the Mexican corpus. See CORPUS.md section 4."""
    from . import db
    from .ingest.mexican import build, write_index

    index = Path(args.from_index) if args.from_index else None
    if index and not index.exists():
        print(f"no such index: {index}")
        return 1
    print("building the Mexican corpus" + (f" from {index}" if index else " from scratch"))
    stats = build(from_index=index, skip_scan=args.skip_scan)
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"  {key:<{width}}  {value}")
    if args.write_index:
        n = write_index(db.client(), args.write_index)
        print(f"  wrote {n} document ranges to {args.write_index}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtext",
        description="Hybrid retrieval over a film dialogue corpus in ClickHouse.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the database and tables").set_defaults(func=cmd_init_db)
    sub.add_parser("status", help="show corpus and configuration state").set_defaults(func=cmd_status)
    sub.add_parser("embed-schema", help="embed column descriptions for schema retrieval").set_defaults(func=cmd_embed_schema)

    p_fetch = sub.add_parser("fetch", help="download the third-party corpora (not redistributed here)")
    p_fetch.add_argument("what", nargs="?", default="all",
                         help="all (default), opus, or imdb")
    p_fetch.add_argument("--lang-pair", default="en-es", help="OPUS language pair (default en-es)")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if present")
    p_fetch.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    p_fetch.set_defaults(func=cmd_fetch)

    p_load = sub.add_parser("load", help="load a corpus into ClickHouse")
    p_load.add_argument("--source", default="sample", choices=["sample", "srt"])
    p_load.add_argument("--path", help="directory of .srt files (for --source srt)")
    p_load.add_argument("--title", help="human-readable title (for --source srt)")
    p_load.add_argument("--title-id", help="stable title id, e.g. tt1234567 (for --source srt)")
    p_load.set_defaults(func=cmd_load)

    p_embed = sub.add_parser("embed", help="chunk and embed the loaded corpus")
    p_embed.add_argument("--strategy", default="line", choices=["line", "window"])
    p_embed.add_argument("--window-size", type=int, default=3)
    p_embed.add_argument("--stride", type=int, default=None)
    p_embed.set_defaults(func=cmd_embed)

    p_search = sub.add_parser("search", help="vector search over dialogue")
    p_search.add_argument("question")
    p_search.add_argument("-k", type=int, default=10)
    p_search.add_argument("--strategy", default="line")
    p_search.add_argument("--window-size", type=int, default=1)
    p_search.add_argument("--character", default=None, help="lines this character speaks")
    p_search.add_argument("--involving", default=None, help="lines this character speaks OR is spoken to")
    p_search.add_argument("--season", type=int, default=None)
    p_search.set_defaults(func=cmd_search)

    p_agg = sub.add_parser("aggregate", help="vector search feeding a SQL GROUP BY")
    p_agg.add_argument("question")
    p_agg.add_argument("--group-by", default="season")
    p_agg.add_argument("-k", type=int, default=40)
    p_agg.add_argument("--strategy", default="line")
    p_agg.add_argument("--window-size", type=int, default=1)
    p_agg.add_argument("--character", default=None, help="lines this character speaks")
    p_agg.add_argument("--involving", default=None, help="lines this character speaks OR is spoken to")
    p_agg.add_argument("--max-distance", type=float, default=0.65)
    p_agg.add_argument("--show-sql", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    p_schema = sub.add_parser("schema", help="show the schema slice retrieved for a question")
    p_schema.add_argument("question")
    p_schema.add_argument("-k", type=int, default=12)
    p_schema.set_defaults(func=cmd_schema)

    p_ask = sub.add_parser("ask", help="ask the ADK agent (needs GOOGLE_API_KEY)")
    p_ask.add_argument("question")
    p_ask.add_argument("--model", default=None)
    p_ask.add_argument("--trace", action="store_true", help="show tool calls and verification")
    p_ask.add_argument(
        "--no-mcp",
        action="store_true",
        help="run SQL through the built-in tool instead of the ClickHouse MCP server",
    )
    p_ask.set_defaults(func=cmd_ask)

    p_serve = sub.add_parser("serve", help="run the web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_mx = sub.add_parser(
        "build-mx",
        help="build the Mexican-Spanish corpus from the loaded parallel corpus")
    p_mx.add_argument(
        "--from-index", metavar="TSV",
        help="use shipped document boundaries (data/mx_docs.tsv) instead of deriving them")
    p_mx.add_argument(
        "--skip-scan", action="store_true",
        help="reuse an existing win_scores table instead of rescanning the corpus")
    p_mx.add_argument(
        "--write-index", metavar="TSV",
        help="write the resulting document ranges out as a redistributable TSV")
    p_mx.set_defaults(func=cmd_build_mx)

    p_eval = sub.add_parser("eval", help="recall@k and faithfulness against the golden set")
    p_eval.add_argument("--golden", default=None)
    p_eval.add_argument("--ks", default="1,3,5,10,20")
    p_eval.add_argument("--strategy", default="line")
    p_eval.add_argument("--window-size", type=int, default=1)
    p_eval.add_argument("--with-agent", action="store_true", help="also score end-to-end answer faithfulness (uses Gemini)")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--output", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_sweep = sub.add_parser("sweep", help="chunk strategy x top-k sweep")
    p_sweep.add_argument("--golden", default=None)
    p_sweep.add_argument("--ks", default="1,3,5,10,20")
    p_sweep.add_argument("--strategies", default="line,window3,window5")
    p_sweep.add_argument("--output", default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
