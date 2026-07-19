"""Duplicate-detection benchmark.

`firefox_bugs-combined.csv` maps a bug to the bug it duplicates. Those pairs are
ground truth: querying with bug A should surface bug B. Nothing in the project read
this file before, so no retrieval change could be judged better or worse than the one
before it. This script turns that into Recall@k and MRR.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.chunking import parent_id_from_chunk_id
from src.config import RERANKER_MODEL, RRF_BM25_WEIGHT
from src.retrieval.retriever import Retriever
from src.storage.chroma_store import ChromaStore

DUPLICATES_CSV = Path("dataset/gitbugs/firefox/firefox_bugs-combined.csv")
OUTPUT_DIR = Path("eval_results")
RANKS = (1, 5, 10)
MAX_K = max(RANKS)


def load_pairs(csv_path: Path, corpus_ids: set[str]) -> list[tuple[str, str]]:
    frame = pd.read_csv(csv_path)
    # The column is string-typed and holds values like "1598175.0"; .astype(int) raises.
    frame["Duplicates"] = pd.to_numeric(frame["Duplicates"], errors="coerce")
    frame = frame.dropna(subset=["Duplicates"])

    pairs = []
    for issue_id, duplicate_of in zip(frame["Issue id"], frame["Duplicates"]):
        query_id = str(int(issue_id))
        target_id = str(int(duplicate_of))
        if query_id in corpus_ids and target_id in corpus_ids:
            pairs.append((query_id, target_id))
    return pairs


# SQLite (Chroma's backing store) rejects a single get() over tens of thousands of
# rows with "too many SQL variables". Page by id in batches of this size.
_GET_BATCH = 500


def _iter_collection(store: ChromaStore, include: list[str]):
    """Yield (ids, documents|None, metadatas|None) pages from the collection."""
    all_ids = store.collection.get(include=[])["ids"]
    for start in range(0, len(all_ids), _GET_BATCH):
        batch_ids = all_ids[start : start + _GET_BATCH]
        page = store.collection.get(ids=batch_ids, include=include)
        yield (
            page["ids"],
            page.get("documents"),
            page.get("metadatas"),
        )


def fetch_parent_documents(store: ChromaStore) -> dict[str, str]:
    """Rebuild one query string per parent by joining chunks in chunk_index order.

    Eval queries used to be a single stored document. With chunking, the body lives
    across rows; concatenating restores the long description for encode_query.
    """
    by_parent: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for ids, documents, metadatas in _iter_collection(
        store, include=["documents", "metadatas"]
    ):
        for chunk_id, document, metadata in zip(
            ids, documents or [], metadatas or []
        ):
            metadata = metadata or {}
            parent = str(
                metadata.get("parent_bug_id") or parent_id_from_chunk_id(chunk_id)
            )
            index = int(metadata.get("chunk_index", 0))
            by_parent[parent].append((index, document or ""))

    documents: dict[str, str] = {}
    for parent, pieces in by_parent.items():
        pieces.sort(key=lambda item: item[0])
        if len(pieces) == 1:
            documents[parent] = pieces[0][1]
            continue
        documents[parent] = "\n".join(text for _, text in pieces)
    return documents


def evaluate(limit: int | None, method: str, bm25_weight: float) -> dict:
    store = ChromaStore()
    if store.count() == 0:
        raise SystemExit("Collection is empty. Run scripts/build_vector_index.py --reset")

    reranker = None
    bm25 = None
    if method == "reranked":
        # Imported lazily so a baseline run never loads the cross-encoder.
        from src.reranking.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        print(f"Reranking with {RERANKER_MODEL}")
    elif method == "hybrid":
        # Imported lazily so a baseline run never pulls in bm25s.
        from src.retrieval.bm25_index import BM25Index

        corpus_ids: list[str] = []
        corpus_docs: list[str] = []
        for ids, documents, _ in _iter_collection(store, include=["documents"]):
            corpus_ids.extend(ids)
            corpus_docs.extend(documents or [""] * len(ids))
        bm25 = BM25Index.from_documents(corpus_ids, corpus_docs)
        print(f"Hybrid dense + BM25 over {len(bm25)} documents (bm25_weight={bm25_weight})")

    retriever = Retriever(store, reranker=reranker, bm25=bm25, bm25_weight=bm25_weight)
    corpus_ids = store.existing_parent_ids()

    pairs = load_pairs(DUPLICATES_CSV, corpus_ids)
    print(f"Usable duplicate pairs (both bugs indexed): {len(pairs)}")
    if limit:
        pairs = pairs[:limit]
        print(f"Evaluating a sample of {len(pairs)}")

    query_documents = fetch_parent_documents(store)

    hits = {rank: 0 for rank in RANKS}
    reciprocal_ranks = []
    evaluated = 0

    for index, (query_id, target_id) in enumerate(pairs, start=1):
        document = query_documents.get(query_id)
        if not document:
            continue

        results = retriever.retrieve(
            document,
            top_k=MAX_K,
            min_similarity=-1.0,  # rank the raw list; do not filter
            exclude_ids=[query_id],  # a bug is not its own duplicate
        )
        retrieved_ids = [result.bug_id for result in results]
        evaluated += 1

        if target_id in retrieved_ids:
            position = retrieved_ids.index(target_id) + 1
            reciprocal_ranks.append(1.0 / position)
            for rank in RANKS:
                if position <= rank:
                    hits[rank] += 1
        else:
            reciprocal_ranks.append(0.0)

        if index % 200 == 0:
            print(f"  {index}/{len(pairs)} queries", end="\r")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corpus_size": store.count(),
        "parent_bugs": len(corpus_ids),
        "pairs_evaluated": evaluated,
        "method": method,
        **({"bm25_weight": bm25_weight} if method == "hybrid" else {}),
        **{f"recall@{rank}": hits[rank] / evaluated for rank in RANKS},
        "mrr": sum(reciprocal_ranks) / evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N pairs. 0 evaluates all of them.",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--rerank",
        action="store_true",
        help="Add the cross-encoder second stage. Compare against a plain run over the "
        "same pairs to measure its effect.",
    )
    stage.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse dense + BM25 with RRF. Compare against a plain run over the same "
        "pairs to measure the effect.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=RRF_BM25_WEIGHT,
        help="Weight on the BM25 side of the RRF fusion (dense is 1.0). Only used with "
        f"--hybrid. Default {RRF_BM25_WEIGHT}. Sweep to tune.",
    )
    args = parser.parse_args()

    method = "reranked" if args.rerank else "hybrid" if args.hybrid else "baseline"
    metrics = evaluate(
        limit=args.limit or None, method=method, bm25_weight=args.bm25_weight
    )

    heading = method
    print(f"\n=== Duplicate detection ({heading}) ===")
    for key, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key:18s} {formatted}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = metrics["timestamp"].replace(":", "-")
    output_path = OUTPUT_DIR / f"retrieval_{heading}_{stamp}.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
