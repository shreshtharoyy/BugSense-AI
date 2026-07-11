import argparse
import logging
from pathlib import Path

from src.indexing.indexing_pipeline import IndexingPipeline
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_mapper import DatasetMapper, RowMappingError
from src.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

DATASET_ROOT = Path("dataset/gitbugs")
PROJECTS = ["firefox"]


def mapped_bugs(csv_path: Path, project: str, already_indexed: set[str] | None = None):
    """Yield BugReports, skipping rows that cannot be mapped.

    Without this, one malformed timestamp aborts a 40-minute run and discards
    everything indexed up to that point.
    """
    failures = 0
    resumed = 0
    for row in DatasetLoader.load(csv_path):
        try:
            bug = DatasetMapper.map_row(row=row, project=project)
        except RowMappingError as exc:
            failures += 1
            if failures <= 10:
                logger.warning("Skipping row in %s: %s", project, exc)
            continue
        if already_indexed and bug.bug_id in already_indexed:
            resumed += 1
            continue
        yield bug
    if failures:
        print(f"Skipped {failures} unmappable rows in {project}")
    if resumed:
        print(f"Resumed past {resumed} already-indexed bugs in {project}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BugSense vector index.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before indexing. Required when "
        "switching distance space, which is immutable once a collection exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip bugs already present in the collection. Embedding runs at roughly "
        "3 bug reports per second on CPU, so a full build is long enough to be worth "
        "restarting rather than repeating.",
    )
    args = parser.parse_args()

    if args.reset and args.resume:
        parser.error("--reset wipes the collection, leaving --resume nothing to skip.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # A pre-existing l2 collection would fail the cosine check in ChromaStore.__init__,
    # which is exactly what --reset is here to repair. Skip the check, then drop it.
    store = ChromaStore(require_cosine=not args.reset)
    if args.reset:
        print("Resetting collection")
        store.reset()

    pipeline = IndexingPipeline(store=store)

    already_indexed = store.existing_ids() if args.resume else None
    if already_indexed:
        print(f"Resuming: {len(already_indexed)} bugs already indexed")

    print("Building BugSense vector index")
    total_indexed = 0

    for project in PROJECTS:
        print(f"Indexing {project}")
        csv_path = DATASET_ROOT / project / f"{project}_bugs.csv"

        total_count = DatasetLoader.count(csv_path)
        bugs = mapped_bugs(csv_path, project, already_indexed)

        # The pipeline returns what it actually wrote. total_count is only the CSV row
        # count, which overstates the result once empty and malformed rows are skipped.
        total_indexed += pipeline.index(bugs_generator=bugs, total_count=total_count)

    print(f"Vector index created. Total bugs indexed: {total_indexed}")
    print(f"Collection now holds: {pipeline.store.count()} vectors")


if __name__ == "__main__":
    main()
