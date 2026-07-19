"""Try duplicate detection on real indexed bugs — interactive or one-shot.

Examples:
  python scripts/try_retrieval.py --sample 3
  python scripts/try_retrieval.py --bug-id 1598175
  python scripts/try_retrieval.py --text "Title: crash on startup. Error Log: NS_ERROR_FAILURE"
  python scripts/try_retrieval.py
"""

from __future__ import annotations

import argparse
import random
import re
import textwrap
from pathlib import Path

import pandas as pd

from src.chunking import parent_id_from_chunk_id
from src.config import RRF_BM25_WEIGHT, TOP_K
from src.retrieval.retriever import RetrievedBug, Retriever
from src.storage.chroma_store import ChromaStore

DUPLICATES_CSV = Path("dataset/gitbugs/firefox/firefox_bugs-combined.csv")
_GET_BATCH = 500
_SNIPPET_CHARS = 220


def load_duplicate_targets(corpus_ids: set[str]) -> dict[str, str]:
    """Map query bug id -> labeled duplicate bug id."""
    if not DUPLICATES_CSV.exists():
        return {}

    frame = pd.read_csv(DUPLICATES_CSV)
    frame["Duplicates"] = pd.to_numeric(frame["Duplicates"], errors="coerce")
    frame = frame.dropna(subset=["Duplicates"])

    targets: dict[str, str] = {}
    for issue_id, duplicate_of in zip(frame["Issue id"], frame["Duplicates"]):
        query_id = str(int(issue_id))
        target_id = str(int(duplicate_of))
        if query_id in corpus_ids and target_id in corpus_ids:
            targets[query_id] = target_id
    return targets


def _iter_collection(store: ChromaStore, include: list[str]):
    all_ids = store.collection.get(include=[])["ids"]
    for start in range(0, len(all_ids), _GET_BATCH):
        batch_ids = all_ids[start : start + _GET_BATCH]
        page = store.collection.get(ids=batch_ids, include=include)
        yield page["ids"], page.get("documents"), page.get("metadatas")


def fetch_parent_document(store: ChromaStore, parent_id: str) -> tuple[str, str]:
    """Return (title, full retrieval text) for one parent bug."""
    page = store.collection.get(
        where={"parent_bug_id": parent_id},
        include=["documents", "metadatas"],
    )
    if not page["ids"]:
        # Legacy single-chunk rows may lack parent_bug_id in metadata.
        page = store.collection.get(
            ids=[f"{parent_id}::0"],
            include=["documents", "metadatas"],
        )

    if not page["ids"]:
        return "", ""

    pieces: list[tuple[int, str, str]] = []
    for chunk_id, document, metadata in zip(
        page["ids"],
        page.get("documents") or [],
        page.get("metadatas") or [],
    ):
        metadata = metadata or {}
        index = int(metadata.get("chunk_index", 0))
        title = str(metadata.get("title") or "")
        pieces.append((index, title, document or ""))

    pieces.sort(key=lambda item: item[0])
    title = pieces[0][1] if pieces else ""
    if len(pieces) == 1:
        return title, pieces[0][2]
    return title, "\n".join(text for _, _, text in pieces)


def build_retriever(store: ChromaStore, hybrid: bool) -> Retriever:
    if not hybrid:
        return Retriever(store)

    from src.retrieval.bm25_index import BM25Index

    print("Building BM25 index from Chroma (one-time, ~1-2 min)...", flush=True)
    corpus_ids: list[str] = []
    corpus_docs: list[str] = []
    for ids, documents, _ in _iter_collection(store, include=["documents"]):
        corpus_ids.extend(ids)
        corpus_docs.extend(documents or [""] * len(ids))

    bm25 = BM25Index.from_documents(corpus_ids, corpus_docs)
    return Retriever(store, bm25=bm25, bm25_weight=RRF_BM25_WEIGHT)


def _score_label(result: RetrievedBug, hybrid: bool) -> str:
    if hybrid and result.rrf_score is not None:
        return f"rrf={result.rrf_score:.4f}"
    return f"sim={result.similarity:.4f}"


def _snippet(document: str) -> str:
    text = re.sub(r"\s+", " ", document).strip()
    if len(text) <= _SNIPPET_CHARS:
        return text
    return text[: _SNIPPET_CHARS - 3].rstrip() + "..."


def print_header(store: ChromaStore, hybrid: bool) -> None:
    parents = len(store.existing_parent_ids())
    mode = "hybrid (dense + BM25)" if hybrid else "dense only"
    print()
    print("=" * 72)
    print(" BugSense AI - duplicate finder demo")
    print("=" * 72)
    print(f" Indexed: {parents:,} bugs / {store.count():,} chunk vectors")
    print(f" Retriever: {mode}")
    print("=" * 72)
    print()


def print_query_block(
    query_label: str,
    title: str,
    query_text: str,
    known_duplicate: str | None,
) -> None:
    print(f"Query: {query_label}")
    if title:
        print(f"Title: {title}")
    if known_duplicate:
        print(f"Labeled duplicate (ground truth): #{known_duplicate}")
    print()
    print("Query text (what gets embedded):")
    print(textwrap.indent(textwrap.fill(_snippet(query_text), width=68), "  "))
    print()


def print_results(
    results: list[RetrievedBug],
    hybrid: bool,
    known_duplicate: str | None,
) -> None:
    if not results:
        print("No matches above threshold.\n")
        return

    print(f"Top {len(results)} similar bugs:")
    print("-" * 72)
    for rank, result in enumerate(results, start=1):
        marker = ""
        if known_duplicate and result.bug_id == known_duplicate:
            marker = "  <-- labeled duplicate"

        print(
            f"#{rank:>2}  bug {result.bug_id:<10}  "
            f"{_score_label(result, hybrid):<16}  {result.title[:48]}{marker}"
        )
        if result.resolution:
            print(f"     resolution: {result.resolution}")
        print(f"     {_snippet(result.document)}")
        print()
    print("-" * 72)
    print()


def run_query(
    retriever: Retriever,
    store: ChromaStore,
    *,
    hybrid: bool,
    query_text: str,
    query_label: str,
    title: str = "",
    exclude_id: str | None = None,
    known_duplicate: str | None = None,
    top_k: int,
) -> None:
    print_query_block(query_label, title, query_text, known_duplicate)

    exclude = [exclude_id] if exclude_id else []
    results = retriever.retrieve(
        query_text,
        top_k=top_k,
        min_similarity=-1.0,
        exclude_ids=exclude,
    )
    print_results(results, hybrid, known_duplicate)

    if known_duplicate:
        hit_rank = next(
            (i + 1 for i, r in enumerate(results) if r.bug_id == known_duplicate),
            None,
        )
        if hit_rank:
            print(f"Ground truth found at rank {hit_rank}.")
        else:
            print("Ground truth was not in this top-k list.")
        print()


def _looks_like_bug_id(text: str) -> bool:
    return bool(re.fullmatch(r"\d+", text.strip()))


def interactive_loop(
    retriever: Retriever,
    store: ChromaStore,
    *,
    hybrid: bool,
    corpus_ids: set[str],
    duplicate_map: dict[str, str],
    top_k: int,
) -> None:
    print("Interactive mode.")
    print("  - Enter a bug number to search using that indexed bug")
    print("  - Or paste a title / description / error message")
    print("  - Empty line to quit")
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            break

        if _looks_like_bug_id(raw):
            bug_id = raw
            if bug_id not in corpus_ids:
                print(f"Bug #{bug_id} is not in the index.\n")
                continue
            title, query_text = fetch_parent_document(store, bug_id)
            if not query_text:
                print(f"Could not load text for bug #{bug_id}.\n")
                continue
            run_query(
                retriever,
                store,
                hybrid=hybrid,
                query_text=query_text,
                query_label=f"indexed bug #{bug_id}",
                title=title,
                exclude_id=bug_id,
                known_duplicate=duplicate_map.get(bug_id),
                top_k=top_k,
            )
            continue

        run_query(
            retriever,
            store,
            hybrid=hybrid,
            query_text=raw,
            query_label="custom text",
            top_k=top_k,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bug-id", help="Query using an indexed bug's full text")
    parser.add_argument("--text", help="Query with free-form bug text")
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Run N random labeled duplicate pairs (shows ground truth)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of results (default {TOP_K})",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Dense-only retrieval (default is champion hybrid)",
    )
    args = parser.parse_args()

    store = ChromaStore()
    if store.count() == 0:
        raise SystemExit(
            "Vector index is empty. Run: python scripts/build_vector_index.py --reset"
        )

    hybrid = not args.dense
    retriever = build_retriever(store, hybrid=hybrid)
    corpus_ids = store.existing_parent_ids()
    duplicate_map = load_duplicate_targets(corpus_ids)

    print_header(store, hybrid)

    if args.bug_id:
        bug_id = args.bug_id.strip()
        if bug_id not in corpus_ids:
            raise SystemExit(f"Bug #{bug_id} is not in the index.")
        title, query_text = fetch_parent_document(store, bug_id)
        if not query_text:
            raise SystemExit(f"Could not load text for bug #{bug_id}.")
        run_query(
            retriever,
            store,
            hybrid=hybrid,
            query_text=query_text,
            query_label=f"indexed bug #{bug_id}",
            title=title,
            exclude_id=bug_id,
            known_duplicate=duplicate_map.get(bug_id),
            top_k=args.top_k,
        )
        return

    if args.text:
        run_query(
            retriever,
            store,
            hybrid=hybrid,
            query_text=args.text,
            query_label="custom text",
            top_k=args.top_k,
        )
        return

    if args.sample:
        if not duplicate_map:
            raise SystemExit(f"No labeled pairs found at {DUPLICATES_CSV}")

        pairs = list(duplicate_map.items())
        random.shuffle(pairs)
        for bug_id, target_id in pairs[: args.sample]:
            title, query_text = fetch_parent_document(store, bug_id)
            if not query_text:
                continue
            run_query(
                retriever,
                store,
                hybrid=hybrid,
                query_text=query_text,
                query_label=f"indexed bug #{bug_id}",
                title=title,
                exclude_id=bug_id,
                known_duplicate=target_id,
                top_k=args.top_k,
            )
        return

    interactive_loop(
        retriever,
        store,
        hybrid=hybrid,
        corpus_ids=corpus_ids,
        duplicate_map=duplicate_map,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
